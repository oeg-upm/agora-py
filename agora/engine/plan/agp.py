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
import re
import traceback
from StringIO import StringIO
from collections import namedtuple
from urlparse import urlparse

import networkx as nx
from networkx.algorithms.isomorphism import DiGraphMatcher
from rdflib import ConjunctiveGraph, BNode
from rdflib import Graph
from rdflib import Variable

__author__ = 'Fernando Serena'

log = logging.getLogger('agora.engine.plan.agp')


def extend_uri(uri, prefixes):
    # type: (str, dict) -> str
    if ':' in uri:
        prefix_parts = uri.split(':')
        if len(prefix_parts) == 2 and prefix_parts[0] in prefixes:
            return prefixes[prefix_parts[0]] + prefix_parts[1]

    return uri.lstrip('<').rstrip('>')


def is_variable(arg):
    # type: (str) -> bool
    return arg.startswith('?')


def is_uri(uri, prefixes):
    # type: (str, dict) -> bool
    if uri.startswith('<') and uri.endswith('>'):
        uri = uri.lstrip('<').rstrip('>')
        parse = urlparse(uri, allow_fragments=True)
        return bool(len(parse.scheme))
    if ':' in uri:
        prefix_parts = uri.split(':')
        return len(prefix_parts) == 2 and prefix_parts[0] in prefixes

    return False


class TP(namedtuple('TP', "s p o")):
    @classmethod
    def _make(cls, iterable, new=tuple.__new__, len=len):
        return new(cls, iterable)

    def __repr__(self):
        # type: () -> str
        def elm_to_string(elm):
            return elm.n3()

        strings = map(elm_to_string, [self.s, self.p, self.o])
        return '{} {} {}'.format(*strings)

    @staticmethod
    def from_string(st, prefixes=None, graph=None):
        # type: (str, dict, Graph) -> TP
        st = st.strip()
        if graph is None:
            graph = Graph()
        ttl = ''

        if prefixes is not None:
            for prefix, uri in prefixes.items():
                ttl += '@prefix {}: <{}> .\n\n'.format(prefix, uri)

        try:
            graph.parse(StringIO(ttl + '\n{} .\n'.format(st)), format='turtle')
            s, p, o = list(graph.triples((None, None, None))).pop()

            return TP._make([s, p, o])
        except Exception:
            traceback.print_exc()
            raise TypeError('Bad TP arguments: {}'.format(st))


class AGP(set):
    def __init__(self, s=(), prefixes=None):
        super(AGP, self).__init__(s)
        self.__prefixes = prefixes or {}
        self.__graph = ConjunctiveGraph()

    @property
    def prefixes(self):
        # type: () -> dict
        return self.__prefixes

    @property
    def wire(self):
        # type: () -> nx.DiGraph
        """
        Creates a graph from the graph pattern
        :return: The graph (networkx)
        """
        g = nx.DiGraph()
        for s, p, o in self:
            edge_data = {'link': p}
            g.add_node(s)
            if isinstance(o, Variable):
                g.add_node(o)
            else:
                g.add_node(o, filter=o)
                edge_data['to'] = o
            g.add_edge(s, o, **edge_data)

        return g

    @property
    def roots(self):
        # type: () -> iter
        def filter_root(x):
            r = x[1] == min_in
            if min_in > 0:
                r = r and x[0] in cycle_elms
            return r

        w = self.wire
        cycle_elms = list(nx.simple_cycles(w))
        if cycle_elms:
            cycle_elms = set.union(*map(lambda x: set(x), list(nx.simple_cycles(w))))
        in_deg = list(w.in_degree())
        min_in = min(map(lambda x: x[1], in_deg)) if in_deg else 0
        roots = map(lambda x: x[0], filter(filter_root, list(w.in_degree())))
        return roots

    def __nodify(self, elm, variables):
        if isinstance(elm, Variable):
            if elm not in variables:
                elm_node = BNode('?' + str(elm))
                variables[elm] = elm_node
            return variables[elm]
        else:
            return elm

    def __iter__(self):
        for x in super(AGP, self).__iter__():
            yield x if isinstance(x, TP) else TP.from_string(x, prefixes=self.__prefixes)

    @property
    def graph(self):
        # type: () -> ConjunctiveGraph
        if not self.__graph:
            for prefix in self.__prefixes:
                self.__graph.bind(prefix, self.__prefixes[prefix])
            variables = {}

            nxg = nx.Graph()
            for (s, p, o) in self:
                nxg.add_nodes_from([s, o])
                nxg.add_edge(s, o)

            contexts = dict([(str(index), c) for (index, c) in enumerate(nx.connected_components(nxg))])

            for (s, p, o) in self:
                s_node = self.__nodify(s, variables)
                o_node = self.__nodify(o, variables)
                p_node = self.__nodify(p, variables)

                context = None
                for uid in contexts:
                    if s in contexts[uid]:
                        context = str(uid)

                if context is not None:
                    self.__graph.get_context(context).add((s_node, p_node, o_node))

        return self.__graph

    @staticmethod
    def from_string(st, prefixes):
        # type: (str, dict) -> AGP
        gp = None
        if st.startswith('{') and st.endswith('}'):
            st = st.replace('{', '').replace('}', '').strip()
            tps = re.split('\. ', st)
            tps = map(lambda x: x.strip().strip('.'), filter(lambda y: y != '', tps))
            gp = AGP(prefixes=prefixes)
            for tp in tps:
                gp.add(TP.from_string(tp, gp.prefixes))
        return gp

    def get_tp_context(self, (s, p, o)):
        # type: (tuple) -> str
        return str(list(self.graph.contexts((s, p, o))).pop().identifier)

    def __repr__(self):
        # type: () -> str
        return '{ %s }' % ' . '.join([str(tp) for tp in self])

    def mapping(self, other):
        # type: (AGP) -> dict
        """
        :return: If there is any, the mapping with another graph pattern
        """
        if not isinstance(other, AGP):
            return {}

        my_wire = self.wire
        others_wire = other.wire

        def __node_match(n1, n2):
            return n1 == n2

        def __edge_match(e1, e2):
            return e1 == e2

        matcher = DiGraphMatcher(my_wire, others_wire, node_match=__node_match, edge_match=__edge_match)
        mapping = list(matcher.isomorphisms_iter())
        if len(mapping) > 0:
            return mapping.pop()
        else:
            return dict()

    def __eq__(self, other):
        # type: (AGP) ->  bool
        """
        Two graph patterns are equal if they are isomorphic**
        """
        if not isinstance(other, AGP):
            return super(AGP, self).__eq__(other)

        mapping = self.mapping(other)
        return mapping is not None
