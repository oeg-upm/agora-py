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
from threading import RLock, Thread, Event, Lock
from xml.sax import SAXParseException

import networkx as nx
from agora.collector.http import get_resource_ttl, RDF_MIMES, http_get
from agora.engine.plan.graph import AGORA
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait
from rdflib import ConjunctiveGraph, RDF, URIRef
from rdflib import Graph
from rdflib import Literal
from rdflib import RDFS
from rdflib import Variable

pool = ThreadPoolExecutor(max_workers=multiprocessing.cpu_count())

__author__ = 'Fernando Serena'

log = logging.getLogger('agora.collector.execution')


class StopException(Exception):
    pass


class FilterTree(set):
    def __init__(self):
        super(FilterTree, self).__init__()
        self.variables = set([])
        self.__lock = Lock()

    def add_variable(self, v):
        self.variables.add(v)

    def filter(self, resource, space, variable):
        with self.__lock:
            # if variable in self.variables:
            self.add((resource, space, variable))

    def is_filtered(self, resource, space, variable):
        with self.__lock:
            # if variable in self.variables:
            return (resource, space, variable) in self
            # return False


class PlanGraph(nx.DiGraph):
    def __init__(self, plan, ss):
        # type: (Graph, SSWrapper) -> PlanGraph
        super(PlanGraph, self).__init__()
        self.__data = {}

        node_patterns = {}
        for (node, _, tp) in plan.triples((None, AGORA.byPattern, None)):
            if node not in node_patterns:
                node_patterns[node] = []
            node_patterns[node].append(tp)
        for node in node_patterns:
            spaces = set([])
            patterns = set([])
            for tp in node_patterns[node]:
                tp_wrapper = ss.patterns[tp]
                spaces.add(tp_wrapper.defined_by)
                patterns.add(tp_wrapper)
            self.add_node(node, spaces=spaces, byPattern=patterns)
            self.__build_from(plan, node, spaces=spaces)

    def __build_from(self, plan, node, **kwargs):
        pred_nodes = list(plan.subjects(AGORA.next, node))
        seeds = set([]) if pred_nodes else plan.objects(node, AGORA.hasSeed)
        self.add_node(node, seeds=seeds, **kwargs)
        for pred in pred_nodes:
            links = list(plan.objects(subject=node, predicate=AGORA.onProperty))
            expected_types = list(plan.objects(subject=node, predicate=AGORA.expectedType))
            link = links.pop() if links else None
            self.add_edge(pred, node, onProperty=link, expectedType=expected_types)
            self.__build_from(plan, pred, **kwargs)

    def add_node(self, n, attr_dict=None, **attr):
        if n not in self.__data:
            self.__data[n] = dict()
        self.__data[n].update(attr)
        for at in attr:
            if isinstance(attr[at], dict):
                self.__data[n][at].update(attr[at])
        super(PlanGraph, self).add_node(n, attr_dict=attr_dict, **self.__data[n])

    def get_node_data(self, node):
        return self.__data[node]


class TPWrapper(object):
    def __init__(self, plan, node):
        # type: (Graph, object) -> TPWrapper
        self.__defined_by = list(plan.subjects(AGORA.definedBy, node)).pop()
        self.__node = node

        predicate = list(plan.objects(node, predicate=AGORA.predicate)).pop()
        self.object_node = list(plan.objects(node, predicate=AGORA.object)).pop()
        self.subject_node = list(plan.objects(node, predicate=AGORA.subject)).pop()
        if isinstance(self.object_node, URIRef):
            object = self.object_node
        elif (self.object_node, RDF.type, AGORA.Literal) in plan:
            object = list(plan.objects(self.object_node, AGORA.value)).pop()
        else:  # It is a variable
            object = Variable(list(plan.objects(self.object_node, RDFS.label)).pop().toPython())

        if isinstance(self.subject_node, URIRef):
            subject = self.subject_node
        else:  # It is a variable
            subject = Variable(list(plan.objects(self.subject_node, RDFS.label)).pop().toPython())

        try:
            self.__check = list(plan.objects(node, predicate=AGORA.checkType)).pop().toPython()
        except IndexError:
            self.__check = True

        self.s = subject
        self.p = predicate
        self.o = object

    @property
    def defined_by(self):
        return self.__defined_by

    @property
    def node(self):
        return self.__node

    @property
    def check(self):
        return self.__check


class SSWrapper(object):
    def __init__(self, plan):
        # type: (Graph) -> SSWrapper
        self.__plan = plan
        self.__spaces = {}
        self.__nodes = {}
        self.__pattern_space = {}
        self.__filter_trees = {}
        for space in self.__plan.subjects(RDF.type, AGORA.SearchSpace):
            self.__spaces[space] = set([])
            self.__filter_trees[space] = []
        tp_nodes = list(self.__plan.subjects(RDF.type, AGORA.TriplePattern))
        self.__patterns = {}
        filter_roots = set([])
        for tp in tp_nodes:
            tp_wrapper = TPWrapper(plan, tp)
            self.__patterns[tp] = tp_wrapper
            self.__nodes[tp_wrapper] = tp
            space = tp_wrapper.defined_by
            self.__spaces[space].add(tp)
            self.__pattern_space[tp_wrapper] = space

            if not isinstance(tp_wrapper.o, Variable):
                filter_roots.add(tp_wrapper)

        for root_tp in filter_roots:
            filter_tree = FilterTree()
            self.__filter_trees[root_tp.defined_by].append(filter_tree)
            self.__build_filter_tree(filter_tree, root_tp)

    def __build_filter_tree(self, fp, tp):
        if isinstance(tp.s, Variable):
            fp.add_variable(tp.s)
            next_tps = list(self.__plan.subjects(AGORA.object, tp.subject_node))
            for tp_node in next_tps:
                tp = self.patterns[tp_node]
                self.__build_filter_tree(fp, tp)

    @property
    def spaces(self):
        return iter(self.__spaces.keys())

    @property
    def patterns(self):
        return self.__patterns

    def space_patterns(self, space):
        # type: (BNode) -> iter
        return self.__spaces[space]

    def filter_trees(self, space):
        return self.__filter_trees[space]


class PlanWrapper(object):
    def __init__(self, plan):
        self.__ss = SSWrapper(plan)
        self.__graph = PlanGraph(plan, self.__ss)

    @property
    def graph(self):
        return self.__graph

    @property
    def roots(self):
        return [(node, data) for (node, data) in self.__graph.nodes(data=True) if not self.__graph.predecessors(node)]

    def successors(self, node):
        # type: (BNode) -> iter

        def filter_weight((n, n_data, edge_data)):
            weight = 2
            if edge_data.get('onProperty', None) is not None:
                weight = 1
            for tp in n_data.get('byPattern', []):
                if isinstance(tp.o, Literal):
                    weight = 0
                    break
            return weight

        suc = [(suc, self.__graph.get_node_data(suc), self.__graph.get_edge_data(node, suc)) for suc in
               self.__graph.successors(node)]
        return sorted(suc, key=lambda x: filter_weight(x))

    def filter(self, resource, space, variable):
        for f in self.__ss.filter_trees(space):
            f.filter(resource, space, variable)

    def is_filtered(self, resource, space, variable=None):
        filters = self.__ss.filter_trees(space)
        if filters:
            for f in self.__ss.filter_trees(space):
                if variable is None:
                    if not any([f.is_filtered(resource, space, v) for v in f.variables]):
                        return False
                else:
                    if not f.is_filtered(resource, space, variable):
                        return False
            return True
        return False


def parse_rdf(graph, content, format):
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


class PlanExecutor(object):
    def __init__(self, plan):

        self.__wrapper = PlanWrapper(plan)
        self.__plan = plan
        self.__resource_lock = RLock()
        self.__locks = {}
        self.__completed = False
        self.__last_success_format = None
        self.__last_iteration_ts = dt.now()
        self.__fragment_ttl = sys.maxint
        self.__last_ttl_ts = None

    @property
    def ttl(self):
        return self.__fragment_ttl

    def resource_lock(self, uri):
        with self.__resource_lock:
            if uri not in self.__locks:
                self.__locks[uri] = Lock()
            return self.__locks[uri]

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
                               loader=None):

        if workers is None:
            workers = multiprocessing.cpu_count()

        fragment_queue = Queue.Queue(maxsize=queue_size)
        workers_queue = Queue.Queue(maxsize=workers)

        if stop_event is None:
            stop_event = Event()

        if loader is None:
            loader = http_get

        def __create_graph():
            if cache is None:
                return ConjunctiveGraph()
            else:
                return cache.create(conjunctive=True)

        def __release_graph(g):
            if cache is not None:
                cache.release(g)
            elif g is not None:
                try:
                    g.remove((None, None, None))
                    g.close()
                except:
                    pass

        def __open_graph(gid, loader, format):
            if cache is None:
                result = loader(gid, format)
                if not isinstance(result, bool):
                    content, headers = result
                    if not isinstance(content, Graph):
                        g = ConjunctiveGraph()
                        parse_rdf(g, content, format)
                    else:
                        g = content

                    ttl = get_resource_ttl(headers)
                    return g, ttl if ttl is not None else 0
                return result
            else:
                return cache.create(gid=gid, loader=loader, format=format)

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
                    resource = __open_graph(uri, loader=loader, format=parse_format)
                    if isinstance(resource, bool):
                        return resource

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
                        __release_graph(g)

            with self.resource_lock(uri):
                if tg.get_context(uri):
                    return

                for fmt in sorted(RDF_MIMES.keys(), key=lambda x: x != self.__last_success_format):
                    if treat_resource_content(fmt):
                        self.__last_success_format = fmt
                        break

        def __check_expected_types(seed, tree_graph, types):
            if not types:
                return True
            for _, _, t in tree_graph.triples((seed, RDF.type, None)):
                if t in types:
                    return True
            return False

        def __process_link_seed(seed, tree_graph, link, next_seeds, expected_types=None):
            __check_stop()
            try:
                __dereference_uri(tree_graph, seed)
                if __check_expected_types(seed, tree_graph, expected_types):
                    seed_pattern_objects = [o for p, o in tree_graph.predicate_objects(subject=seed) if p == link]
                    next_seeds.update(seed_pattern_objects)
            except Exception as e:
                traceback.print_exc()
                log.warning(e.message)

        def __process_pattern_link_seed(seed, tree_graph, pattern_link, expected_types=None):
            __check_stop()
            try:
                __dereference_uri(tree_graph, seed)
            except:
                pass
            seed_pattern_objects = set([])
            if __check_expected_types(seed, tree_graph, expected_types):
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
            fragment_queue.put(quad, timeout=queue_wait)

        def __follow_node(node, seed, tree_graph):
            candidates = []
            try:
                for n, n_data, e_data in self.__wrapper.successors(node):
                    expected_types = e_data.get('expectedType', None)
                    patterns = n_data.get('byPattern', [])
                    on_property = e_data.get('onProperty', None)

                    seed_variables = set([])
                    for space in n_data['spaces']:
                        for tp in patterns:
                            seed_variables.add(tp.s)

                            if space != tp.defined_by:
                                continue

                            if self.__wrapper.is_filtered(seed, space, tp.s):
                                continue

                            if isinstance(tp.s, URIRef) and seed != tp.s:
                                continue

                            if tp.p != RDF.type:
                                try:
                                    sobs = list(__process_pattern_link_seed(seed, tree_graph, tp.p,
                                                                            expected_types=expected_types))

                                    # TODO: This may not apply when considering OPTIONAL support
                                    if not isinstance(tp.o, Variable) and not sobs:
                                        self.__wrapper.filter(seed, space, tp.s)

                                    if not sobs:
                                        self.__wrapper.filter(seed, space, tp.s)

                                    for object in sobs:
                                        __check_stop()

                                        if not isinstance(tp.o, Variable) and object.n3() != tp.o.n3():
                                            self.__wrapper.filter(seed, space, tp.s)
                                        else:
                                            candidate = (tp, seed, object)
                                            candidates.append(candidate)
                                except AttributeError as e:
                                    log.warning('Trying to find {} objects of {}: {}'.format(tp.p, seed, e.message))
                            else:
                                __dereference_uri(tree_graph, seed)
                                try:
                                    seed_objects = list(tree_graph.objects(subject=seed, predicate=on_property))
                                    for seed_object in seed_objects:
                                        # In some cases, it is necessary to verify the type of the seed
                                        put_quad = True
                                        if tp.check:
                                            __dereference_uri(tree_graph, seed_object)
                                            types = list(
                                                tree_graph.objects(subject=seed_object, predicate=RDF.type))
                                            if tp.o not in types:
                                                put_quad = False

                                        if put_quad:
                                            candidates.append((tp, seed_object, tp.o))
                                except AttributeError as e:
                                    log.warning(
                                        'Trying to find {} objects of {}: {}'.format(on_property, seed, e.message))

                        next_seeds = set([])

                        if on_property is not None:
                            if seed_variables:
                                if all([self.__wrapper.is_filtered(seed, space, v) for v in seed_variables]):
                                    continue

                            __process_link_seed(seed, tree_graph, on_property, next_seeds,
                                                expected_types=expected_types)

                        try:
                            threads = []
                            for s in next_seeds:
                                __check_stop()
                                try:
                                    workers_queue.put_nowait(s)
                                    future = pool.submit(__follow_node, n, s, tree_graph)
                                    threads.append(future)
                                except Queue.Full:
                                    # If all threads are busy...I'll do it myself
                                    __follow_node(n, s, tree_graph)

                                if len(threads) >= workers:
                                    wait(threads)
                                    [(workers_queue.get_nowait(), workers_queue.task_done()) for _ in threads]
                                    threads = []

                            wait(threads)
                            [(workers_queue.get_nowait(), workers_queue.task_done()) for _ in threads]
                            next_seeds.clear()
                        except (IndexError, KeyError):
                            traceback.print_exc()

                predicate_pass = {}
                quads = set([])
                for (tp, seed, object) in candidates:
                    quad = (tp.node, seed, tp.p, object)
                    passing = not self.__wrapper.is_filtered(seed, space, tp.s)
                    passing = passing and not self.__wrapper.is_filtered(object, space, tp.o)
                    if not passing:
                        if tp.p not in predicate_pass:
                            predicate_pass[tp.p] = False
                        continue

                    predicate_pass[tp.p] = True
                    quads.add(quad)

                if all(predicate_pass.values()):
                    for q in quads:
                        __put_quad_in_queue(q)

            except Queue.Full:
                stop_event.set()
            except Exception as e:
                traceback.print_exc()
                log.error(e.message)

        def get_fragment_triples():
            """
            Iterate over all search trees and yield relevant triples
            :return:
            """

            def execute_plan():
                futures = []

                for tree, data in self.__wrapper.roots:
                    # Prepare an dedicated graph for the current tree and a set of type triples (?s a Concept)
                    # to be evaluated retrospectively
                    tree_graph = __create_graph()

                    try:
                        # Get all seeds of the current tree
                        seeds = set(data['seeds'])
                        # Check if the tree root is a pattern node and in that case, adds a type triple to the
                        # respective set
                        tree_patterns = data.get('byPattern', [])
                        root_type_candidates = set([])
                        for tp in [tp for tp in tree_patterns if tp.p == RDF.type]:
                            [root_type_candidates.add((tp, seed, tp.o)) for seed in seeds]

                        for seed in seeds:
                            futures.append(pool.submit(__follow_node, tree, seed, tree_graph))
                        wait(futures)

                        for (tp, seed, type) in root_type_candidates:
                            passing = not self.__wrapper.is_filtered(seed, tp.defined_by, tp.s)
                            if not passing:
                                continue

                            __put_quad_in_queue((tp.node, seed, RDF.type, type))

                    finally:
                        __release_graph(tree_graph)

                self.__completed = True
                __update_fragment_ttl()

            thread = Thread(target=execute_plan)
            thread.daemon = True
            thread.start()

            while not self.__completed or fragment_queue.not_empty:

                try:
                    (t, s, p, o) = fragment_queue.get(timeout=0.001)
                    fragment_queue.task_done()
                    yield (t, s, p, o)
                except Queue.Empty:
                    if self.__completed:
                        break
                self.__last_iteration_ts = dt.now()
            thread.join()

        return {'generator': get_fragment_triples(), 'prefixes': self.__plan.namespaces(),
                'plan': self.__plan}
