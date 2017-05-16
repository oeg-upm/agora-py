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
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait
from rdflib import BNode
from rdflib import ConjunctiveGraph, RDF, URIRef
from rdflib import Graph
from rdflib import RDFS
from rdflib import Variable
from rdflib.plugins.sparql.algebra import translateQuery
from rdflib.plugins.sparql.parser import expandUnicodeEscapes, Query
from rdflib.plugins.sparql.sparql import QueryContext
from shortuuid import uuid

from agora.collector.http import get_resource_ttl, RDF_MIMES, http_get
from agora.engine.plan.graph import AGORA

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
        self.graph = nx.DiGraph()

    def add_tp(self, tp):
        self.graph.add_edge(tp.s, tp.o)

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
            cycle_start = list(plan.objects(subject=node, predicate=AGORA.isCycleStartOf))
            link = links.pop() if links else None
            self.add_edge(pred, node, onProperty=link, expectedType=expected_types, cycleStart=cycle_start)
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

    def __repr__(self):
        def elm_to_string(elm):
            return elm.n3()

        strings = map(elm_to_string, [self.s, self.p, self.o])
        return '{} {} {}'.format(*strings)


class Cycle(nx.DiGraph):
    def __init__(self, plan, node):
        super(Cycle, self).__init__()
        self.__root_types = None
        self.__root_node = node
        self.__build_from(plan, node)

    def __build_from(self, plan, node, **kwargs):
        next_nodes = list(plan.objects(node, AGORA.next))
        self.add_node(node, **kwargs)
        for nxt in next_nodes:
            links = list(plan.objects(subject=nxt, predicate=AGORA.onProperty))
            expected_types = list(plan.objects(subject=node, predicate=AGORA.expectedType))
            if self.__root_types is None:
                self.__root_types = expected_types
            link = links.pop() if links else None
            self.add_edge(node, nxt, onProperty=link, expectedType=expected_types)
            self.__build_from(plan, nxt, **kwargs)

    @property
    def root_types(self):
        return frozenset(self.__root_types)

    @property
    def root_node(self):
        return self.__root_node


class SSWrapper(object):
    def __init__(self, plan):
        self.__plan = plan
        self.__spaces = {}
        self.__nodes = {}
        self.__cycles = {}
        self.__pattern_space = {}
        self.__filter_trees = {}
        self.__tp_graph = nx.DiGraph()
        for space in self.__plan.subjects(RDF.type, AGORA.SearchSpace):
            self.__spaces[space] = set([])
            self.__filter_trees[space] = []
        tp_nodes = list(self.__plan.subjects(RDF.type, AGORA.TriplePattern))
        self.__patterns = {}
        filter_roots = set([])

        cycle_nodes = list(self.__plan.subjects(RDF.type, AGORA.Cycle))
        for c_node in cycle_nodes:
            cycle = Cycle(plan, c_node)
            for type in cycle.root_types:
                if type not in self.__cycles:
                    self.__cycles[type] = set([])
                self.__cycles[type].add(cycle)

        for tp in tp_nodes:
            tp_wrapper = TPWrapper(plan, tp)
            self.__patterns[tp] = tp_wrapper
            self.__nodes[tp_wrapper] = tp
            space = tp_wrapper.defined_by
            self.__spaces[space].add(tp)
            self.__pattern_space[tp_wrapper] = space
            self.__tp_graph.add_edge(tp_wrapper.s, tp_wrapper.o)

            if not isinstance(tp_wrapper.o, Variable) or not isinstance(tp_wrapper.s, Variable):
                filter_roots.add(tp_wrapper)

        for root_tp in filter_roots:
            filter_tree = FilterTree()
            self.__filter_trees[root_tp.defined_by].append(filter_tree)
            self.__build_filter_tree(filter_tree, root_tp)
            filter_tree.add_variable(root_tp.o)

    def __build_filter_tree(self, fp, tp):
        if isinstance(tp.s, Variable):
            fp.add_tp(tp)
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

    @property
    def cycles(self):
        return self.__cycles

    def space_patterns(self, space):
        # type: (BNode) -> iter
        return self.__spaces[space]

    def filter_trees(self, space):
        return self.__filter_trees[space]

    def filtered_vars(self, space):
        return reduce(lambda x, y: x.union(y.variables), self.__filter_trees[space], set([]))

    def filter_var(self, tp, v):
        if tp.defined_by not in self.__filter_trees or not self.__filter_trees[tp.defined_by]:
            ft = FilterTree()
            self.__filter_trees[tp.defined_by] = [ft]
        for ft in self.__filter_trees[tp.defined_by]:
            self.__build_filter_tree(ft, tp)
            ft.add_variable(v)

    @property
    def tp_graph(self):
        return self.__tp_graph

    def clear_filters(self):
        for space_list in self.__filter_trees.values():
            for ft in space_list:
                ft.clear()


class PlanWrapper(object):
    def __filter_cycle_extensions(self, cycle, (u, v, data)):
        return cycle.root_node in data['cycleStart']

    def __init__(self, plan):
        self.__ss = SSWrapper(plan)
        self.__graph = PlanGraph(plan, self.__ss)

        cycles = reduce(lambda x, y: y.union(x), self.__ss.cycles.values(), set([]))

        cycles_dict = {c: [c_data for (_, _, c_data) in c.edges_iter(c.root_node, data=True)] for c in cycles}
        ext_steps = {}
        for c in cycles:
            ext_steps[c] = filter(lambda e: self.__filter_cycle_extensions(c, e),
                                  self.__graph.edges(data=True))
        for c in cycles:
            c_edges = cycles_dict[c]

            for (u, v, data) in ext_steps[c]:
                for i, c_edge in enumerate(c_edges):
                    source = u if i == 0 else self.__clone_node(u)
                    dest = u if i == len(c_edges) - 1 else self.__clone_node(u)
                    existing_data = self.__graph.get_edge_data(source, dest)

                    on_property = {c_edge['onProperty']}
                    if existing_data:
                        on_property.update(existing_data['onProperty'])
                    self.__graph.add_edge(source, dest, expectedType=c_edge['expectedType'],
                                          onProperty=on_property, cycle=True)

    def __clone_node(self, node):
        n = uuid()
        n_data = self.__graph.get_node_data(node).copy()
        n_data['spaces'] = []
        self.__graph.add_node(n, n_data)
        return n

    @property
    def graph(self):
        return self.__graph

    @property
    def roots(self):
        def __predecessors(x):
            in_edges = self.__graph.in_edges(x, data=True)
            result = bool(filter(lambda (u, v, d): not d.get('cycle', False), in_edges))
            return result

        return [(node, data) for (node, data) in self.__graph.nodes(data=True) if not __predecessors(node)]

    @property
    def patterns(self):
        return self.__ss.patterns.values()

    def cycles_for(self, ty):
        return self.__ss.cycles.get(ty, [])

    def successors(self, node):
        # type: (BNode) -> iter

        def filter_weight((n, n_data, edge_data)):
            weight = 2
            if edge_data.get('cycle', False):
                weight = 5000
            elif edge_data.get('onProperty', None) is not None:
                weight = 1
            aggr_dist = 1000
            patterns = n_data.get('byPattern', [])
            if patterns:
                weight = -aggr_dist
                for tp in patterns:
                    filtered_vars = list(self.__ss.filtered_vars(tp.defined_by))
                    for ft in self.__ss.filter_trees(tp.defined_by):
                        nodes = ft.graph.nodes()
                        if tp.o in ft.graph:
                            for v in filtered_vars:
                                if tp.o in nodes and v in nodes:
                                    try:
                                        dist = nx.shortest_path(ft.graph, tp.o, v)
                                        aggr_dist = min(aggr_dist, len(dist))
                                    except nx.NetworkXNoPath:
                                        pass
                                    except nx.NetworkXError as e:
                                        print e.message
                    weight += aggr_dist

            return weight

        suc = []
        for (u, v, data) in self.__graph.edges_iter(node, data=True):
            on_property = data.get('onProperty')
            if isinstance(on_property, set):
                for prop in on_property:
                    prop_data = data.copy()
                    prop_data['onProperty'] = prop
                    suc.append((v, self.__graph.get_node_data(v), prop_data))
            else:
                suc.append((v, self.__graph.get_node_data(v), data))

        sorted_suc = sorted(suc, key=lambda x: filter_weight(x))

        return sorted_suc

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

    def under_filter(self, space):
        return any(filter(lambda x: len(x) > 0, self.__ss.filter_trees(space)))

    def filter_var(self, tp, v):
        self.__ss.filter_var(tp, v)

    def connected(self, v1, v2):
        try:
            nx.shortest_path(self.__ss.tp_graph, v1, v2)
            return True
        except nx.NetworkXNoPath:
            return False

    def clear_filters(self):
        self.__ss.clear_filters()


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
                               loader=None, filters=None, follow_cycles=False):

        if workers is None:
            workers = multiprocessing.cpu_count()

        fragment_queue = Queue.Queue(maxsize=queue_size)
        workers_queue = Queue.Queue(maxsize=workers)

        fragment = set([])

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
            self.__n_derefs += 1
            if cache is None:
                result = loader(gid, format)
                if result is None and loader != http_get:
                    result = http_get(gid, format)
                if isinstance(result, tuple):
                    # not isinstance(result, bool):
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

        def __process_link_seed(seed, tree_graph, link, next_seeds):
            __check_stop()
            try:
                __dereference_uri(tree_graph, seed)
                seed_pattern_objects = [o for p, o in tree_graph.predicate_objects(subject=seed) if p == link]
                next_seeds.update(seed_pattern_objects)
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

        def __any_parent_filtered(space, parent, tp):
            for (ps, link, cycle, vars) in parent:
                if not cycle:
                    for v in vars:
                        if self.__wrapper.is_filtered(ps, space, v) and self.__wrapper.connected(v, tp.s):
                            return True
            return False

        def __follow_node(node, seed, tree_graph, parent=None):
            if parent is None:
                parent = []
            candidates = set([])
            quads = set([])
            predicate_pass = {}
            seed_variables = set([])
            try:
                if (node, seed) in self.__node_seeds:
                    # print 'already visited: {} with {}'.format(node, seed)
                    return
                self.__node_seeds.add((node, seed))

                for n, n_data, e_data in self.__wrapper.successors(node):
                    expected_types = e_data.get('expectedType', None)
                    patterns = n_data.get('byPattern', [])
                    on_property = e_data.get('onProperty', None)
                    on_cycle = e_data.get('cycle', False)

                    if not follow_cycles and on_cycle:
                        continue

                    spaces = [] if on_cycle else n_data['spaces']
                    next_seeds = set([])
                    for space in spaces:
                        next_seeds.clear()

                        last_tp = None
                        sobs = []

                        for tp in sorted(patterns, key=lambda x: __tp_weight(x), reverse=True):
                            predicate_pass[tp.p] = True

                            if (space, on_property, tp.p, seed) in self.__node_seeds:
                                # print 'already visited: {} {} for {}'.format(seed, on_property, tp.p)
                                continue
                            self.__node_seeds.add((space, on_property, tp.p, seed))

                            if __any_parent_filtered(space, parent, tp):
                                continue

                            seed_variables.add(tp.s)

                            if space != tp.defined_by:
                                continue

                            if self.__wrapper.is_filtered(seed, space, tp.s):
                                continue

                            if not on_cycle and isinstance(tp.s, URIRef) and seed != tp.s:
                                self.__wrapper.filter(seed, space, tp.s)
                                continue
                            else:  # tp.s is a Variable
                                if tp.s in var_filters:
                                    for var_f in var_filters[tp.s]:
                                        context = QueryContext()
                                        context[tp.o] = seed
                                        passing = var_f.expr.eval(context) if hasattr(var_f.expr, 'eval') else bool(
                                            seed.toPython())
                                        if not passing:
                                            self.__wrapper.filter(seed, space, tp.s)
                                            continue

                            if tp.p != RDF.type:
                                try:
                                    if not last_tp or last_tp.p != tp.p:
                                        sobs = list(__process_pattern_link_seed(seed, tree_graph, tp.p))

                                    # TODO: This may not apply when considering OPTIONAL support
                                    if not isinstance(tp.o, Variable) and not sobs:
                                        predicate_pass[tp.p] = False
                                        self.__wrapper.filter(seed, space, tp.s)

                                    if not sobs and not on_cycle:
                                        predicate_pass[tp.p] = False
                                        self.__wrapper.filter_var(tp, tp.s)
                                        self.__wrapper.filter(seed, space, tp.s)

                                    filtered = False
                                    obs_candidates = set([])
                                    for object in sobs:
                                        __check_stop()

                                        if not isinstance(tp.o, Variable):
                                            if object.n3() != tp.o.n3():
                                                filtered = True
                                                # break
                                        else:
                                            if tp.o in var_filters:
                                                for var_f in var_filters[tp.o]:
                                                    context = QueryContext()
                                                    context[tp.o] = object
                                                    passing = var_f.expr.eval(context) if hasattr(var_f.expr,
                                                                                                  'eval') else bool(
                                                        object.toPython())
                                                    if not passing:
                                                        self.__wrapper.filter(seed, space, tp.s)
                                                        filtered = True
                                                        # break

                                        if not filtered:
                                            candidate = (tp, seed, object)
                                            obs_candidates.add(candidate)

                                    candidates.update(obs_candidates)

                                    if not next_seeds and tp.p == on_property:
                                        next_seeds = set(sobs)

                                except AttributeError as e:
                                    log.warning('Trying to find {} objects of {}: {}'.format(tp.p, seed, e.message))
                            else:
                                put_quad = True
                                if tp.check:
                                    __dereference_uri(tree_graph, seed)
                                    types = set(
                                        tree_graph.objects(subject=seed, predicate=RDF.type))
                                    if not set.intersection(types, expected_types):
                                        put_quad = False
                                if put_quad:
                                    candidates.add((tp, seed, tp.o))
                            last_tp = tp

                    new_parent = parent
                    if on_property is not None:
                        if seed_variables:
                            if not on_cycle and all(
                                    [self.__wrapper.is_filtered(seed, space, v) for v in seed_variables]):
                                continue

                            new_parent = parent[:]
                            new_parent.append((seed, on_property, on_cycle, seed_variables))

                        if not next_seeds:
                            __process_link_seed(seed, tree_graph, on_property, next_seeds)

                        # if next_seeds and on_cycle:
                        #     print 'entering cycle: {} -> {} -> {}'.format(seed, on_property, next_seeds)

                    try:
                        threads = []
                        for s in next_seeds:
                            __check_stop()
                            try:
                                workers_queue.put_nowait(s)
                                future = pool.submit(__follow_node, n, s, tree_graph, parent=new_parent)
                                threads.append(future)
                            except Queue.Full:
                                # If all threads are busy...I'll do it myself
                                __follow_node(n, s, tree_graph, parent=new_parent)

                            if len(threads) >= workers:
                                wait(threads)
                                [(workers_queue.get_nowait(), workers_queue.task_done()) for _ in threads]
                                threads = []

                        wait(threads)
                        [(workers_queue.get_nowait(), workers_queue.task_done()) for _ in threads]
                        next_seeds.clear()
                    except (IndexError, KeyError):
                        traceback.print_exc()

                    for (tp, cand_seed, object) in candidates:
                        quad = (tp, cand_seed, tp.p, object)
                        passing = not self.__wrapper.is_filtered(cand_seed, space, tp.s)
                        passing = passing and not self.__wrapper.is_filtered(object, space, tp.o)
                        if not passing:
                            if tp.p not in predicate_pass:
                                predicate_pass[tp.p] = False
                            continue

                        predicate_pass[tp.p] = True
                        quads.add(quad)

                    try:
                        if candidates and (not predicate_pass or not all(predicate_pass.values())):
                            quads.clear()
                            break
                    finally:
                        candidates.clear()

                if all(predicate_pass.values()):
                    for q in quads:
                        __put_quad_in_queue(q)
                elif predicate_pass:
                    for v in seed_variables:
                        self.__wrapper.filter(seed, space, v)
                    quads.clear()

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
                for tree, data in self.__wrapper.roots:
                    # Prepare an dedicated graph for the current tree and a set of type triples (?s a Concept)
                    # to be evaluated retrospectively
                    tree_graph = __create_graph()
                    self.__node_seeds.clear()
                    self.__wrapper.clear_filters()

                    try:
                        # Get all seeds of the current tree
                        seeds = set(data['seeds'])
                        # Check if the tree root is a pattern node and in that case, adds a type triple to the
                        # respective set
                        tree_patterns = data.get('byPattern', [])
                        root_type_candidates = set([])
                        for tp in [tp for tp in tree_patterns if tp.p == RDF.type]:
                            [root_type_candidates.add((tp, seed, tp.o)) for seed in seeds]

                        threads = []
                        for seed in seeds:
                            try:
                                workers_queue.put_nowait(seed)
                                future = pool.submit(__follow_node, tree, seed, tree_graph)
                                threads.append(future)
                            except Queue.Full:
                                # If all threads are busy...I'll do it myself
                                __follow_node(tree, seed, tree_graph)

                            if len(threads) >= workers:
                                wait(threads)
                                [(workers_queue.get_nowait(), workers_queue.task_done()) for _ in threads]
                                threads = []

                        wait(threads)
                        [(workers_queue.get_nowait(), workers_queue.task_done()) for _ in threads]

                        for (tp, seed, type) in root_type_candidates:
                            passing = not self.__wrapper.is_filtered(seed, tp.defined_by, tp.s)
                            if not passing:
                                continue

                            __put_quad_in_queue((tp, seed, RDF.type, type))

                    finally:
                        __release_graph(tree_graph)

                self.__completed = True
                __update_fragment_ttl()

            thread = Thread(target=execute_plan)
            thread.daemon = True
            thread.start()

            while not self.__completed or not fragment_queue.empty():
                try:
                    q = fragment_queue.get(timeout=0.01)
                    fragment_queue.task_done()
                    yield q
                except Queue.Empty:
                    if self.__completed:
                        break
                self.__last_iteration_ts = dt.now()
            thread.join()

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
