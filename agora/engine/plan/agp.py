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
from collections import namedtuple
from urlparse import urlparse

import networkx as nx
import re
from rdflib import ConjunctiveGraph, URIRef, BNode, RDF, Literal
from rdflib import Variable

__author__ = 'Fernando Serena'

log = logging.getLogger('agora.engine.plan.agp')


def extend_uri(uri, prefixes):
    if ':' in uri:
        prefix_parts = uri.split(':')
        if len(prefix_parts) == 2 and prefix_parts[0] in prefixes:
            return prefixes[prefix_parts[0]] + prefix_parts[1]

    return uri.lstrip('<').rstrip('>')


def is_variable(arg):
    return arg.startswith('?')


def is_uri(uri, prefixes):
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
    def _make(cls, iterable, new=tuple.__new__, len=len, prefixes=None):
        def transform_elm(elm):
            if is_variable(elm):
                return Variable(elm)
            elif is_uri(elm, prefixes):
                return URIRef(extend_uri(elm, prefixes))
                # return URIRef(elm.lstrip('<').rstrip('>'))
            elif elm == 'a':
                return RDF.type
            else:
                datatype = None
                if '^^' in elm:
                    elm, datatype = elm.split('^^')
                elm = elm.lstrip('"').rstrip('"')
                if datatype is not None:
                    datatype = URIRef(extend_uri(datatype, prefixes))
                return Literal(elm, datatype=datatype)

        if prefixes is None:
            prefixes = []
        res = filter(lambda x: x, map(transform_elm, iterable))
        if len(res) == 3:
            if not (isinstance(res[0], Literal) or isinstance(res[1], Literal)):
                return new(cls, res)

        raise TypeError('Bad TP arguments: {}'.format(iterable))

    def __repr__(self):
        def elm_to_string(elm):
            if isinstance(elm, URIRef):
                if elm == RDF.type:
                    return 'a'
                return '<%s>' % elm

            return str(elm)

        strings = map(elm_to_string, [self.s, self.p, self.o])
        return '{} {} {}'.format(*strings)

    @staticmethod
    def from_string(st, prefixes):
        if st.endswith('"'):
            parts = [st[st.find('"'):]]
            st = st.replace(parts[0], '').rstrip()
            parts = st.split(" ") + parts
        else:
            parts = st.split(' ')
        return TP._make(parts, prefixes=prefixes)


class AgoraGP(object):
    def __init__(self, prefixes):
        self._tps = []
        self.__prefixes = prefixes

    @property
    def triple_patterns(self):
        return self._tps

    @property
    def prefixes(self):
        return self.__prefixes

    @property
    def graph(self):
        g = ConjunctiveGraph()
        for prefix in self.__prefixes:
            g.bind(prefix, self.__prefixes[prefix])
        variables = {}

        def nodify(elm):
            if isinstance(elm, Variable):
                if not (elm in variables):
                    elm_node = BNode('?' + str(elm))
                    variables[elm] = elm_node
                return variables[elm]
            else:
                return elm

        nxg = nx.Graph()
        for (s, p, o) in self._tps:
            nxg.add_nodes_from([s, o])
            nxg.add_edge(s, o)

        contexts = dict([(str(index), c) for (index, c) in enumerate(nx.connected_components(nxg))])

        for (s, p, o) in self._tps:
            s_node = nodify(s)
            o_node = nodify(o)
            p_node = nodify(p)

            context = None
            for uid in contexts:
                if s in contexts[uid]:
                    context = str(uid)

            g.get_context(context).add((s_node, p_node, o_node))

        return g

    @staticmethod
    def from_string(st, prefixes):
        gp = None
        if st.startswith('{') and st.endswith('}'):
            st = st.replace('{', '').replace('}', '').strip()
            tps = re.split('\. ', st)
            tps = map(lambda x: x.strip().strip('.'), filter(lambda y: y != '', tps))
            gp = AgoraGP(prefixes)
            for tp in tps:
                gp.triple_patterns.append(TP.from_string(tp, gp.prefixes))
        return gp

    def get_tp_context(self, (s, p, o)):
        return str(list(self.graph.contexts((s, p, o))).pop().identifier)

    def __repr__(self):
        tp_strings = map(lambda x: str(x), self._tps)
        return '{ %s}' % reduce(lambda x, y: (x + '%s . ' % str(y)), tp_strings, '')
