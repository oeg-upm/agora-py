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

import logging
from threading import Lock

import networkx as nx
from rdflib import BNode
from rdflib import RDF, URIRef
from rdflib import RDFS
from rdflib import Variable
from shortuuid import uuid

from agora.engine.plan.graph import AGORA

__author__ = 'Fernando Serena'

log = logging.getLogger('agora.collector.plan')


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
            self.add((resource, space, variable))

    def is_filtered(self, resource, space, variable):
        with self.__lock:
            return (resource, space, variable) in self


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
            check_l = list(plan.objects(node, AGORA.checkType))
            check = check_l.pop().toPython() if check_l else False
            for tp in node_patterns[node]:
                tp_wrapper = ss.patterns[tp]
                spaces.add(tp_wrapper.defined_by)
                patterns.add(tp_wrapper)
            self.add_node(node, spaces=spaces, byPattern=patterns, check=check)
            self.__build_from(plan, node, spaces=spaces)

    def __build_from(self, plan, node, **kwargs):
        pred_nodes = list(plan.subjects(AGORA.next, node))
        seeds = set([]) if pred_nodes else plan.objects(node, AGORA.hasSeed)
        self.add_node(node, seeds=seeds, **kwargs)
        for pred in pred_nodes:
            links = list(plan.objects(subject=node, predicate=AGORA.onProperty))
            type_hierarchy = list(plan.objects(subject=node, predicate=AGORA.typeHierarchy))
            type_hierarchy = type_hierarchy.pop() if type_hierarchy else False
            expected_types = list(set(plan.objects(subject=node, predicate=AGORA.expectedType)))
            cycle_start = list(plan.objects(subject=node, predicate=AGORA.isCycleStartOf))
            link = links.pop() if links else None
            self.add_edge(pred, node, onProperty=link, expectedType=expected_types, cycleStart=cycle_start,
                          typeHierarchy=type_hierarchy)
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

        self.s = subject
        self.p = predicate
        self.o = object

    @property
    def defined_by(self):
        return self.__defined_by

    @property
    def node(self):
        return self.__node

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
            type_hierarchy = list(plan.objects(subject=node, predicate=AGORA.typeHierarchy))
            type_hierarchy = type_hierarchy.pop() if type_hierarchy else False
            expected_types = list(plan.objects(subject=node, predicate=AGORA.expectedType))
            if self.__root_types is None:
                self.__root_types = expected_types
            link = links.pop() if links else None
            self.add_edge(node, nxt, onProperty=link, expectedType=expected_types, typeHierarchy=type_hierarchy)
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

    def __build_filter_tree(self, fp, tp, trace=None):
        if isinstance(tp.s, Variable):
            if trace is None:
                trace = []
            fp.add_tp(tp)
            next_tps = list(self.__plan.subjects(AGORA.object, tp.subject_node))
            for tp_node in next_tps:
                tp = self.patterns[tp_node]
                if tp not in trace:
                    trace.append(tp)
                    self.__build_filter_tree(fp, tp, trace)

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
    @staticmethod
    def __filter_cycle_extensions(cycle, (u, v, data)):
        return cycle.root_node in data['cycleStart']

    def __init__(self, plan):
        self.__ss = SSWrapper(plan)
        self.__graph = PlanGraph(plan, self.__ss)
        self.__node_succesors = {}
        self.__node_spaces = {}
        self.__pattern_successors = {}
        self.__link_successors = {}
        self.__node_links = {}
        self.__node_patterns = {}
        self.__inverses = {}

        for r in plan.query("""
                SELECT DISTINCT ?p ?inv WHERE {
                    ?inv owl:inverseOf ?p
                }"""):
            self.__inverses[r.p] = r.inv

        cycles = reduce(lambda x, y: y.union(x), self.__ss.cycles.values(), set([]))

        ext_steps = {}
        for c in cycles:
            ext_steps[c] = filter(lambda e: self.__filter_cycle_extensions(c, e),
                                  self.__graph.edges(data=True))
        source_cycle_map = {}
        for c in cycles:
            c_edges = list(c.edges())

            for (u, v, data) in ext_steps[c]:
                if u not in source_cycle_map:
                    source_cycle_map[u] = set()
                if c not in source_cycle_map[u]:
                    source_cycle_map[u].add(c)
                else:
                    continue

                c_root = c.root_node
                last_node = None
                for i in range(len(c_edges)):
                    cu, cv, c_edge = list(c.edges_iter(c_root, data=True)).pop()
                    source = u if i == 0 else last_node
                    dest = u if i == len(c_edges) - 1 else self.__clone_node(c, cv, self.__graph.get_node_data(u))
                    last_node = dest

                    c_root = cv
                    existing_data = self.__graph.get_edge_data(source, dest)

                    on_property = {c_edge['onProperty']}
                    if existing_data:
                        on_property.update(existing_data['onProperty'])
                    self.__graph.add_edge(source, dest, expectedType=c_edge['expectedType'],
                                          onProperty=on_property, cycle=True, typeHierarchy=c_edge['typeHierarchy'])

    def __clone_node(self, c, node, data):
        n = BNode(uuid())
        n_data = data.copy()
        if 'seeds' in n_data:
            del n_data['seeds']
        self.__graph.add_node(n, n_data)
        return n

    @property
    def inverses(self):
        return self.__inverses

    @property
    def graph(self):
        return self.__graph

    @property
    def roots(self):
        return [(node, data) for (node, data) in self.__graph.nodes(data=True) if data.get('seeds', False)]

    @property
    def patterns(self):
        return self.__ss.patterns.values()

    def cycles_for(self, ty):
        return self.__ss.cycles.get(ty, [])

    def successors(self, node):
        # type: (BNode) -> iter

        if node in self.__node_succesors:
            return self.__node_succesors[node]

        def filter_weight((n, n_data, edge_data)):
            weight = 2
            if edge_data.get('cycle', False):
                weight = 5000
            elif n_data.get('byPattern', False):
                weight = -5000
            elif edge_data.get('onProperty', None) is not None:
                aggr_dist = 1000
                patterns = on_property_nodes.get((n, edge_data.get('onProperty', None)), [])
                if patterns:
                    weight = -aggr_dist
                    dist_modifier = None
                    for tp in patterns:
                        filtered_vars = list(self.__ss.filtered_vars(tp.defined_by))
                        for ft in self.__ss.filter_trees(tp.defined_by):
                            nodes = ft.graph.nodes()
                            if tp.o in ft.graph:
                                for v in filtered_vars:
                                    if tp.o in nodes and v in nodes:
                                        try:
                                            dist = nx.shortest_path(ft.graph, tp.o, v)
                                            dist_modifier = min(dist_modifier or 1000, len(dist))
                                        except nx.NetworkXNoPath:
                                            pass
                                        except nx.NetworkXError as e:
                                            print e.message
                        weight -= (dist_modifier or 0)

            return weight

        suc = []
        on_property_nodes = {}
        all_suc = self.__graph.edges_iter(node, data=True)
        for (u, v, data) in sorted(all_suc, key=lambda x: x[2].get('onProperty') is None):
            on_property = data.get('onProperty')

            by_pattern = self.__graph.get_node_data(v).get('byPattern', None)
            if by_pattern:
                for tp in by_pattern:
                    for n in filter(lambda x: x[1] == tp.p, on_property_nodes):
                        on_property_nodes[n].add(tp)

            if isinstance(on_property, set):
                for prop in on_property:
                    on_property_nodes[(v, prop)] = set()
                    prop_data = data.copy()
                    prop_data['onProperty'] = prop
                    suc.append((v, self.__graph.get_node_data(v), prop_data))
            else:
                if on_property:
                    on_property_nodes[(v, on_property)] = set()
                suc.append((v, self.__graph.get_node_data(v), data))

        sorted_suc = sorted(suc, key=lambda x: filter_weight(x))

        self.__node_succesors[node] = sorted_suc
        return sorted_suc

    def node_spaces(self, node):
        if node not in self.__node_spaces:
            successors = self.successors(node)
            spaces = set(
                reduce(lambda x, y: set.union(x, y), map(lambda x: x[1].get('spaces', set()), successors),
                       set()))
            self.__node_spaces[node] = spaces
        return self.__node_spaces[node]

    def pattern_successors(self, node):
        if node not in self.__pattern_successors:
            successors = self.successors(node)
            pattern_succ = filter(lambda (n, n_data, e_data): n_data.get('byPattern', []), successors)
            self.__pattern_successors[node] = pattern_succ
        return self.__pattern_successors[node]

    def link_successors(self, node):
        if node not in self.__link_successors:
            successors = self.successors(node)
            link_succ = filter(lambda (n, n_data, e_data): e_data.get(
                'onProperty', None) and not e_data.get(
                'cycle', False), successors)
            # link_succ = filter(lambda (n, n_data, e_data): not n_data.get('byPattern', None) and e_data.get(
            #     'onProperty', None) and not e_data.get(
            #     'cycle', False), successors)
            self.__link_successors[node] = link_succ
        return self.__link_successors[node]

    def node_links(self, node):
        if node not in self.__node_links:
            link_succ = self.link_successors(node)
            links = set(
                reduce(lambda x, y: x.union({y}), map(lambda x: x[2]['onProperty'], link_succ),
                       set()))
            self.__node_links[node] = links
        return self.__node_links[node]

    def node_patterns(self, node):
        if node not in self.__node_patterns:
            pattern_succ = self.pattern_successors(node)
            patterns = reduce(lambda x, y: x.union(y), map(lambda x: x[1]['byPattern'], pattern_succ),
                              set())
            patterns = map(lambda x: x.p, patterns)
            self.__node_patterns[node] = patterns
        return self.__node_patterns[node]

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
