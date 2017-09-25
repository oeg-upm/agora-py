"""
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#
  Ontology Engineering Group
        http://www.oeg-upm.net/
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#
  Copyright (C) 2016 Ontology Engineering Group.
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#
  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

            http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#
"""

import Queue
import gc
import logging
import multiprocessing
import sys
import traceback
from _bsddb import DBNotFoundError
from datetime import datetime as dt, datetime
from threading import RLock, Thread, Lock
from xml.sax import SAXParseException

from concurrent.futures import ThreadPoolExecutor, wait
from rdflib import ConjunctiveGraph, RDF, URIRef
from rdflib import Graph
from rdflib import Variable
from rdflib.plugins.sparql.algebra import translateQuery
from rdflib.plugins.sparql.parser import expandUnicodeEscapes, Query
from rdflib.plugins.sparql.sparql import QueryContext

from agora.collector.http import get_resource_ttl, RDF_MIMES, http_get
from agora.collector.plan import PlanWrapper
from agora.engine.utils import stopped

__author__ = 'Fernando Serena'

log = logging.getLogger('agora.collector.execution')


class QueueInterceptor(object):
    def __init__(self, parent):
        self.queue = Queue.Queue()
        self.parent = parent

    def put(self, elm):
        self.parent.put(elm)
        self.queue.put(elm)

    def get(self):
        return self.queue.get()


class StopException(Exception):
    pass


def parse_rdf(graph, content, format, headers):
    content_type = headers.get('Content-Type')
    if content_type:
        for f, mime in RDF_MIMES.items():
            if mime in content_type:
                format = f
    try:
        graph.parse(content, format=format)
    except SyntaxError:
        traceback.print_exc()
        return False

    except ValueError:
        traceback.print_exc()
        return False
    except DBNotFoundError:
        # Ignore this exception... it is raised due to a stupid problem with prefixes
        traceback.print_exc()
        return True
    except SAXParseException:
        traceback.print_exc()
        return False
    except Exception:
        traceback.print_exc()
        return True


def _create_graph(cache=None):
    if cache is None:
        return ConjunctiveGraph()
    else:
        return cache.create(conjunctive=True)


def _release_graph(g, cache=None):
    try:
        if cache is not None:
            cache.release(g)
        elif g is not None:
            g.remove((None, None, None))
            g.close()
    except:
        pass


def _open_graph(gid, loader, format, cache=None):
    if cache is None:
        result = loader(gid, format)
        if result is None and loader != http_get:
            result = http_get(gid, format)
        if isinstance(result, tuple):
            content, headers = result
            if not isinstance(content, Graph):
                g = ConjunctiveGraph()
                parse_rdf(g, content, format, headers)
            else:
                g = content

            ttl = get_resource_ttl(headers)
            return g, ttl if ttl is not None else 0
        return result
    else:
        return cache.create(gid=gid, loader=loader, format=format)


def _follow_in_breadth(n, next_seeds, tree_graph, workers, follow, pool, parent=None, queue=None, cycle=False):
    try:
        threads = []
        for s in next_seeds:
            try:
                workers.put_nowait(s)
                future = pool.submit(follow, n, s, tree_graph, parent=parent, queue=queue)
                threads.append(future)
            except Queue.Full:
                # If all threads are busy...I'll do it myself
                follow(n, s, tree_graph, parent=parent, queue=queue, cycle=cycle)

        if len(threads) >= workers:
            wait(threads)
            [(workers.get_nowait(), workers.task_done()) for _ in threads]
            threads = []

        wait(threads)
        [(workers.get_nowait(), workers.task_done()) for _ in threads]
        next_seeds.clear()
    except (IndexError, KeyError):
        traceback.print_exc()


class PlanExecutor(object):
    pool = ThreadPoolExecutor(max_workers=(2 * multiprocessing.cpu_count()) + 1)

    def __init__(self, plan):

        self.__wrapper = PlanWrapper(plan)
        self.__plan = plan
        self.__resource_lock = RLock()
        self.__tp_lock = RLock()
        self.__node_lock = RLock()
        self.__locks = {}
        self.__completed = False
        self.__aborted = False
        self.__last_success_format = None
        self.__last_iteration_ts = dt.now()
        self.__fragment_ttl = sys.maxint
        self.__n_derefs = 0
        self.__last_ttl_ts = None
        self.__node_seeds = set([])

    @property
    def ttl(self):
        return self.__fragment_ttl

    @property
    def n_derefs(self):
        return self.__n_derefs

    def resource_lock(self, uri):
        with self.__resource_lock:
            if uri not in self.__locks:
                self.__locks[uri] = Lock()
            return self.__locks[uri]

    def tp_lock(self, seed, tp):
        with self.__tp_lock:
            if (seed, tp) not in self.__locks:
                self.__locks[(seed, tp)] = Lock()
            return self.__locks[(seed, tp)]

    def node_lock(self, node, seed):
        with self.__node_lock:
            if (node, seed) not in self.__locks:
                self.__locks[(node, seed)] = Lock()
            return self.__locks[(node, seed)]

    def get_fragment(self, **kwargs):
        """
        Return a complete fragment.
        :param gp:
        :return:
        """
        gen, namespaces, plan = self.get_fragment_generator(**kwargs)
        graph = ConjunctiveGraph()
        [graph.bind(prefix, u) for (prefix, u) in namespaces]
        [graph.add((s, p, o)) for (_, s, p, o) in gen]

        return graph

    def get_fragment_generator(self, workers=None, stop_event=None, queue_wait=None, queue_size=100, cache=None,
                               loader=None, filters=None, follow_cycles=True, type_strict=True):

        if workers is None:
            workers = multiprocessing.cpu_count()

        fragment_queue = Queue.Queue(maxsize=queue_size)
        workers_queue = Queue.Queue(maxsize=workers)

        fragment = set([])

        if stop_event is None:
            stop_event = stopped

        if loader is None:
            loader = http_get

        def __update_fragment_ttl():
            now = datetime.utcnow()
            if self.__last_ttl_ts is not None:
                self.__fragment_ttl -= (now - self.__last_ttl_ts).total_seconds()
                self.__fragment_ttl = max(self.__fragment_ttl, 0)
            self.__last_ttl_ts = now

        def __dereference_uri(tg, uri):

            if not isinstance(uri, URIRef):
                return

            uri = uri.toPython()
            uri = uri.encode('utf-8')

            def treat_resource_content(parse_format):

                try:
                    resource = _open_graph(uri, loader=loader, format=parse_format, cache=cache)
                    self.__n_derefs += 1
                    if isinstance(resource, bool):
                        return resource
                except KeyboardInterrupt:
                    stop_event.set()
                    return
                except Exception, e:
                    traceback.print_exc()
                    log.warn(e.message)
                    return

                g, ttl = resource

                try:
                    __update_fragment_ttl()
                    self.__fragment_ttl = min(self.__fragment_ttl, ttl)
                    tg.get_context(uri).__iadd__(g)
                    return True
                finally:
                    if g is not None:
                        _release_graph(g, cache)

            with self.resource_lock(uri):
                if tg.get_context(uri):
                    return

                for fmt in sorted(RDF_MIMES.keys(), key=lambda x: x != self.__last_success_format):
                    if treat_resource_content(fmt):
                        self.__last_success_format = fmt
                        break

        def __process_link_seed(seed, tree_graph, link, next_seeds):
            __check_stop()
            try:
                __dereference_uri(tree_graph, seed)
                seed_pattern_objects = [o for p, o in tree_graph.predicate_objects(subject=seed) if p == link]
                next_seeds.update(seed_pattern_objects)
            except KeyboardInterrupt:
                stop_event.set()
            except Exception as e:
                traceback.print_exc()
                log.warning(e.message)

        def __process_pattern_link_seed(seed, tree_graph, pattern_link):
            __check_stop()
            try:
                __dereference_uri(tree_graph, seed)
            except:
                pass
            seed_pattern_objects = [o for p, o in tree_graph.predicate_objects(subject=seed) if p == pattern_link]
            return seed_pattern_objects

        def __check_stop():
            if stop_event.isSet():
                gc.collect()
                raise StopException()

        def __put_quad_in_queue(quad):
            if (dt.now() - self.__last_iteration_ts).total_seconds() > 100:
                log.info('Aborted fragment collection!')
                stop_event.set()
            if quad not in fragment:
                fragment.add(quad)
                fragment_queue.put(quad, timeout=queue_wait)

        def __tp_weight(x):
            weight = int(x.s in var_filters) + int(x.o in var_filters)
            return weight

        def __process_candidates(candidates, space):
            quads = set([])
            predicate_pass = {}
            candidates = sorted(candidates, key=lambda x: x[0].p)
            for i in range(len(candidates)):
                (tp, cand_seed, object) = candidates.pop()
                if tp.p not in predicate_pass:
                    predicate_pass[tp.p] = False

                quad = (tp, cand_seed, tp.p, object)
                s_passing = not self.__wrapper.is_filtered(cand_seed, space, tp.s)
                if not s_passing:
                    continue

                o_passing = not self.__wrapper.is_filtered(object, space, tp.o)
                if o_passing:
                    predicate_pass[tp.p] = True
                    quads.add(quad)

            if not all(predicate_pass.values()):
                quads.clear()
            return quads

        def __process_pattern(seed, space, tp, expected_types, check, graph):
            candidates = set([])
            with self.tp_lock(seed, tp):
                self.__wrapper.filter_var(tp, tp.s)

                if self.__wrapper.is_filtered(seed, space, tp.s):
                    return

                if isinstance(tp.s, URIRef) and seed != tp.s:
                    return

                else:  # tp.s is a Variable
                    if tp.s in var_filters:
                        for var_f in var_filters[tp.s]:
                            context = QueryContext()
                            context[tp.o] = seed
                            passing = var_f.expr.eval(context) if hasattr(var_f.expr, 'eval') else bool(
                                seed.toPython())
                            if not passing:
                                return

                if tp.p != RDF.type or isinstance(tp.o, Variable):
                    try:
                        sobs = list(__process_pattern_link_seed(seed, graph, tp.p))

                        # TODO: This may not apply when considering OPTIONAL support
                        if not isinstance(tp.o, Variable) and not sobs:
                            return

                        if not sobs:
                            return

                        obs_candidates = []
                        for object in sobs:
                            filtered = True
                            __check_stop()

                            if not isinstance(tp.o, Variable):
                                if object.n3() == tp.o.n3():
                                    filtered = False
                            else:
                                if tp.o in var_filters:
                                    for var_f in var_filters[tp.o]:
                                        context = QueryContext()
                                        context[tp.o] = object
                                        passing = var_f.expr.eval(context) if hasattr(var_f.expr,
                                                                                      'eval') else bool(
                                            object.toPython())
                                        if passing:
                                            filtered = False
                                else:
                                    filtered = False

                            if not filtered:
                                candidate = (tp, seed, object)
                                obs_candidates.append(candidate)

                        candidates.update(obs_candidates)
                    except AttributeError as e:
                        log.warning('Trying to find {} objects of {}: {}'.format(tp.p, seed, e.message))
                else:
                    if type_strict and check:
                        __dereference_uri(graph, seed)
                        types = set(
                            graph.objects(subject=seed, predicate=RDF.type))
                        if tp.o not in types:
                            # print 'filtering ' + seed + ' for ' + str(tp.s)
                            return

                    candidates.add((tp, seed, tp.o))

            return candidates

        def __process_seed_patterns(seed, p_nodes, graph):
            space_dict = {}
            seed_v_pass = {}

            for n, n_data, e_data in p_nodes:
                check = n_data.get('check', False)
                expected_types = e_data.get('expectedType', set())
                spaces = n_data['spaces']
                for space in spaces:
                    patterns = filter(lambda p: p.defined_by == space, n_data.get('byPattern', []))
                    seed_v = set()
                    candidates = set()
                    ef_patterns = set()
                    space_dict[space] = {
                        'candidates': candidates,
                        'seed_v': seed_v,
                        'patterns': ef_patterns
                    }

                    patterns = sorted(patterns, key=lambda x: __tp_weight(x), reverse=True)
                    for tp in patterns:
                        seed_v.add(tp.s)
                        if tp.s not in seed_v_pass:
                            seed_v_pass[tp.s] = True

                        tp_candidates = __process_pattern(seed, space, tp, expected_types, check, graph)
                        if tp_candidates:
                            candidates.update(tp_candidates)
                            ef_patterns.add(tp)
                        else:
                            seed_v_pass[tp.s] = False
                            self.__wrapper.filter(seed, space, tp.s)

            if not all(seed_v_pass.items()):
                return {}

            for seed_v in filter(lambda x: not seed_v_pass[x], seed_v_pass):
                for s_dict in space_dict.values():
                    if seed_v in s_dict['seed_v']:
                        s_dict['candidates'] = set(filter(lambda x: x[0].s != seed_v, s_dict['candidates']))
                        s_dict['seed_v'].remove(seed_v)

            return space_dict

        def __process_seed_links(seed, predicates, p_links, graph, parent, queue):
            try:
                filter_predicates = False
                for n, n_data, e_data in p_links:
                    on_property = e_data.get('onProperty', None)
                    if filter_predicates and on_property in predicates:
                        continue

                    type_hierarchy = e_data.get('typeHierarchy', False)
                    next_seeds = set()
                    if on_property is not None:
                        new_parent = parent[:]
                        link_queue = QueueInterceptor(queue)
                        __process_link_seed(seed, graph, on_property, next_seeds)
                        n_messages = len(next_seeds)
                        follow_queue = link_queue if on_property in predicates else queue
                        _follow_in_breadth(n, next_seeds, graph, workers_queue, __follow_node, PlanExecutor.pool,
                                           parent=new_parent, queue=follow_queue)

                        if link_queue == follow_queue:
                            found_any = False

                            for i in range(n_messages):
                                child_msg = follow_queue.get()
                                if child_msg:
                                    found_any = found_any or child_msg[1]
                                    if found_any:
                                        break

                            if not found_any and type_hierarchy:
                                filter_predicates = True

                queue.put(None)
            except StopException:
                pass

        def __send_quads(seed, quads, variables, space):
            for q in quads:
                __put_quad_in_queue(q)

        def __notify(queue, seed, found):
            if queue:
                queue.put((seed, found))

        def __follow_node(node, seed, tree_graph, parent=None, queue=None, cycle=False):
            found_triples = False
            notified = False
            try:
                if parent is None:
                    parent = []

                evaluate_and_stop = False
                if (node, seed) in parent:
                    return

                if (node, seed) in self.__node_seeds:
                    if cycle:
                        return
                    evaluate_and_stop = True

                parent = parent[:]
                parent.append((node, seed))

                with self.node_lock(node, seed):
                    try:
                        if (node, seed) in self.__node_seeds:
                            if cycle:
                                return
                            evaluate_and_stop = True
                        else:
                            self.__node_seeds.add((node, seed))

                        try:
                            __dereference_uri(tree_graph, seed)
                        except:
                            pass

                        seed_types = set(tree_graph.objects(seed, RDF.type))

                        successors = list(self.__wrapper.successors(node))
                        spaces = set(self.__wrapper.node_spaces(node))
                        pattern_succ = self.__wrapper.pattern_successors(node)
                        pattern_succ = list(filter(
                            lambda (n, n_data, e_data): set.intersection(seed_types, e_data.get('expectedType', [])),
                            pattern_succ))

                        link_succ = self.__wrapper.link_successors(node)
                        link_succ = list(filter(
                            lambda (n, n_data, e_data): set.intersection(seed_types, e_data.get('expectedType', [])),
                            link_succ))

                        space_dict = __process_seed_patterns(seed, pattern_succ, tree_graph)
                        links = self.__wrapper.node_links(node)
                        pattern_predicates = self.__wrapper.node_patterns(node)

                        for space in spaces:
                            space_link_succ = filter(lambda x: space in x[1].get('spaces'), link_succ)
                            space_pattern_succ = filter(lambda x: space in x[1].get('spaces'), pattern_succ)

                            # Check if links and patterns successors have a common expectedType
                            expected_in_patterns = set(reduce(lambda x, y: x + y,
                                                              map(lambda x: x[2]['expectedType'],
                                                                  space_pattern_succ), []))
                            expected_in_links = set(reduce(lambda x, y: x + y,
                                                           map(lambda x: x[2]['expectedType'],
                                                               space_link_succ), []))

                            if space_pattern_succ and not space_dict.get(space, {}).get('candidates'):
                                val_space_link = []
                                for l in space_link_succ:
                                    l_expected = set(l[2]['expectedType'])
                                    # if expected_in_patterns.difference(l_expected):
                                    if not expected_in_patterns.intersection(l_expected):
                                        val_space_link.append(l)

                                space_link_succ = val_space_link

                            follow_thread = None
                            follow_queue = Queue.Queue()

                            if not evaluate_and_stop and space_link_succ:
                                follow_thread = Thread(target=__process_seed_links,
                                                       args=(
                                                           seed, pattern_predicates, space_link_succ[:], tree_graph,
                                                           parent, follow_queue))
                                follow_thread.start()

                            if space in space_dict:
                                s_dict = space_dict[space]
                                patterns = s_dict['patterns']

                                predicates = set(map(lambda x: x.p, patterns))
                                direct_predicates = set.difference(predicates, links)
                                predicate_pass = {pred: pred in direct_predicates for pred in predicates}

                                wait_for_links = set.intersection(expected_in_links, expected_in_patterns)

                                if predicate_pass and all(predicate_pass.values()):
                                    candidates = s_dict['candidates']
                                    seed_variables = s_dict['seed_v']
                                    quads = __process_candidates(candidates, space)
                                    found_triples = bool(len(quads))
                                    __send_quads(seed, quads, seed_variables, space)
                                elif follow_thread and wait_for_links:
                                    retained_candidates = {}
                                    while True:
                                        child_msg = follow_queue.get()
                                        if not child_msg:
                                            break

                                        child_seed, found = child_msg
                                        if found:
                                            s_dict = space_dict[space]
                                            candidates = s_dict['candidates']
                                            seed_variables = s_dict['seed_v']
                                            child_candidates = filter(lambda x: x[2] == child_seed, candidates)

                                            if all(predicate_pass.values()):
                                                s_dict['candidates'] = set.difference(candidates, child_candidates)
                                                quads = __process_candidates(child_candidates, space)
                                                found_triples = bool(len(quads))
                                                __send_quads(seed, quads, seed_variables, space)
                                                if found_triples and not notified:
                                                    __notify(queue, seed, found_triples)
                                                    notified = True
                                            else:
                                                if space not in retained_candidates:
                                                    retained_candidates[space] = set()

                                                for cc in child_candidates:
                                                    predicate_pass[cc[0].p] = True
                                                    retained_candidates[space].add(cc)

                                                if all(predicate_pass.values()):
                                                    retained = retained_candidates.get(space, set())
                                                    retained.update(
                                                        filter(lambda x: x[0].p in direct_predicates, candidates))
                                                    s_dict['candidates'] = set.difference(candidates, retained)
                                                    quads = __process_candidates(retained, space)
                                                    found_triples = bool(len(quads))
                                                    if found_triples and not notified:
                                                        __notify(queue, seed, found_triples)
                                                        notified = True
                                                    __send_quads(seed, quads, seed_variables, space)

                                                    # if all(predicate_pass.values()):
                                                    #     candidates = s_dict['candidates']
                                                    #     seed_variables = s_dict['seed_v']
                                                    #     quads = __process_candidates(candidates, space)
                                                    #     s_dict['candidates'] = set()
                                                    #     __send_quads(seed, quads, seed_variables, space)

                            if follow_thread:
                                follow_thread.join()

                        if not follow_cycles:
                            return

                        cycle_succ = filter(lambda (n, n_data, e_data): e_data.get('cycle', False), successors)
                        next_seeds = set()
                        for n, n_data, e_data in cycle_succ:
                            p = parent[:]
                            on_property = e_data.get('onProperty', None)
                            __process_link_seed(seed, tree_graph, on_property, next_seeds)
                            next_seeds = set(filter(lambda s: (n, s) not in p, next_seeds))
                            if next_seeds:
                                print 'entering cycle: {} -> {} -> {}'.format(seed, on_property, next_seeds)
                                _follow_in_breadth(n, next_seeds, tree_graph, workers_queue, __follow_node,
                                                   PlanExecutor.pool,
                                                   parent=p, cycle=True)

                    except (Queue.Full, StopException):
                        stop_event.set()
                    except Exception as e:
                        traceback.print_exc()
                        log.error(e.message)
                        raise e
            finally:
                if not notified:
                    __notify(queue, seed, found_triples)

        def get_fragment_triples():
            """
            Iterate over all search trees and yield relevant triples
            :return:
            """

            def execute_plan():
                try:
                    for tree, data in self.__wrapper.roots:
                        # Prepare an dedicated graph for the current tree and a set of type triples (?s a Concept)
                        # to be evaluated retrospectively
                        tree_graph = _create_graph(cache)
                        self.__node_seeds.clear()
                        self.__wrapper.clear_filters()

                        try:
                            # Get all seeds of the current tree
                            seeds = set(data['seeds'])
                            _follow_in_breadth(tree, seeds, tree_graph, workers_queue, __follow_node,
                                               PlanExecutor.pool)
                        except StopException, e:
                            raise e
                        finally:
                            _release_graph(tree_graph, cache)

                    self.__completed = True
                    __update_fragment_ttl()
                except StopException:
                    self.__aborted = True

            thread = Thread(target=execute_plan)
            thread.start()

            while not self.__aborted and (not self.__completed or not fragment_queue.empty()):
                try:
                    q = fragment_queue.get(timeout=0.01)
                    fragment_queue.task_done()
                    yield q
                except Queue.Empty:
                    if self.__completed:
                        break
                except KeyboardInterrupt, e:
                    stop_event.set()
                    PlanExecutor.pool.shutdown(wait=True)
                    raise e

                self.__last_iteration_ts = dt.now()
            thread.join()

            if self.__aborted or not self.__completed:
                log.info('Aborted plan execution!')
                raise StopException('Aborted plan execution')
            else:
                log.info('Finished plan execution!')

        var_filters = {}
        if filters:
            for v in filters:
                var_filters[v] = []
                for f in filters[v]:
                    f = 'SELECT * WHERE { FILTER (%s) }' % f
                    parse = Query.parseString(expandUnicodeEscapes(f), parseAll=True)
                    query = translateQuery(parse)
                    var_filters[v].append(query.algebra.p.p)
                for tp in filter(lambda x: x.s == v or x.o == v, self.__wrapper.patterns):
                    self.__wrapper.filter_var(tp, v)

        return {'generator': get_fragment_triples(), 'prefixes': self.__plan.namespaces(),
                'plan': self.__plan}
