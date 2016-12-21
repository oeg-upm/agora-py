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

import networkx as nx
from rdflib import BNode
from rdflib import Graph
from rdflib import RDF
from rdflib import Variable
from rdflib.plugins.sparql.sparql import AlreadyBound
from rdflib.plugins.sparql.sparql import QueryContext

from agora.engine.plan import AGP
from agora.engine.plan.graph import AGORA

__author__ = 'Fernando Serena'


class Context(object):
    def __init__(self, **kwargs):
        self.map = kwargs

    def __setitem__(self, key, value):
        if isinstance(key, Variable):
            self.map[key] = value

    def __getitem__(self, item):
        return self.map.get(item, None)

    def __eq__(self, other):
        if isinstance(other, Context):
            return self.map == other.map
        return False

    def __contains__(self, item):
        return item in self.map

    def __repr__(self):
        s = ''
        for k in sorted(self.map.keys()):
            if isinstance(k, Variable):
                s += u'{}={} '.format(k.toPython(), self.map[k].toPython())
        return s

    @property
    def variables(self):
        return set([k for k in self.map.keys() if isinstance(k, Variable)])

    def issubset(self, other):
        for k in self.map.keys():
            if k not in other.map or self.map[k] != other.map[k]:
                return False

        return True


class ContextCollection(set):
    def __init__(self):
        super(ContextCollection, self).__init__()

    def __contains__(self, other):
        # type: (Context) -> bool
        for c in self:
            if c.map == other.map:
                return True
        return False


def __base_generator(agp, ctx, fragment):
    # type: (AGP, QueryContext, iter) -> iter
    tp_single_vars = [(tp, tp.s if isinstance(tp.s, Variable) else tp.o) for tp in agp if
                      isinstance(tp.s, Variable) != isinstance(tp.o, Variable)]

    wire = agp.wire
    ignore_tps = [str(tp) for (tp, v) in tp_single_vars if len(wire.neighbors(v)) > 1]
    for tp, ss, _, so in fragment:
        if str(tp) in ignore_tps:
            continue
        kwargs = {}
        if isinstance(tp.o, Variable):
            ctx_o = ctx[tp.o]
            if ctx_o is not None and so != ctx_o:
                continue
            kwargs[tp.o] = so
        if isinstance(tp.s, Variable):
            ctx_s = ctx[tp.s]
            if ctx_s is not None and ss != ctx_s:
                continue
            kwargs[tp.s] = ss

        if kwargs:
            yield Context(**kwargs), tp


def __diff(c1, c2):
    # type: (Context, Context, iter) -> iter
    for k in set(c1.map.keys()).union(set(c2.map.keys())):
        if k in c1 and k in c2 and c1[k] != c2[k]:
            yield k
        elif not (k in c1 and k in c2):
            yield k


def __exploit(c1, c2):
    # type: (Context, Context) -> Context
    new_dict = {v: c1[v] for v in c1.map}
    extend = True
    for v in __diff(c1, c2):
        if c1[v] is None:
            new_dict[v] = c2[v]
        elif c2[v] is not None:
            extend = False
            break

    if extend:
        c = Context(**new_dict)
        return c


def __joins(c, x):
    # type: (Context, Context) -> bool
    any = False
    len_c = len(c.variables)
    intersection = c.variables.intersection(x.variables)
    if len_c - len(intersection) >= 1:
        any = True
        for v in intersection:
            if c[v] != x[v]:
                return False

    return any


def __eval_delta(c, graph, roots, variables):
    # type: (Context, nx.DiGraph) -> iter

    def filter_successor(x):
        return x != c and __joins(c, x) and not set.intersection(set(nx.descendants(graph, x)),
                                                                 set(nx.descendants(graph, c)))

    solutions = ContextCollection()

    root_candidates = reduce(lambda x, y: set.union(x, set(graph.successors(y))), roots, set())
    for root in root_candidates:
        if filter_successor(root):
            inter = __exploit(root, c)
            if inter is not None:
                graph.add_edge(root, inter)
                graph.add_edge(c, inter)
                if len(inter.variables) == len(variables):
                    solutions.add(inter)
                else:
                    pred = filter(lambda x: graph.out_degree(x) > 1, graph.predecessors(inter))
                    for s in __eval_delta(inter, graph, pred, variables):
                        solutions.add(s)

    return solutions


def __query_context(ctx, c):
    # type: (QueryContext, Context) -> QueryContext
    q = ctx.push()
    for k, v in c.map.items():
        try:
            q[k] = v
        except AlreadyBound:
            pass
    return q


def print_ancestors(graph, node, source=None, g=None, bids=None, cids=None):
    if g is None:
        g = Graph()
    if bids is None:
        bids = {}
    if cids is None:
        cids = {}

    if isinstance(node, Variable):
        gn = BNode(node.toPython())
        g.add((gn, RDF.type, AGORA.Variable))
    else:
        if node not in cids:
            cids[node] = BNode('c{}'.format(len(cids)))
        gn = cids[node]
        for v in node.variables:
            bid_tuple = (v, node[v])
            if bid_tuple not in bids:
                bids[bid_tuple] = BNode('b' + str(len(bids)))
            bn = bids[bid_tuple]
            g.add((gn, AGORA.hasBinding, bn))
            g.set((bn, RDF.type, AGORA.Binding))
            g.set((bn, AGORA.forVariable, BNode(v.toPython())))
            g.set((bn, AGORA.withValue, node[v]))

        for pred in graph.predecessors(node):
            print_ancestors(graph, pred, source=gn, g=g, bids=bids, cids=cids)

    if source is not None:
        g.add((source, AGORA.source, gn))
    else:
        g.add((gn, RDF.type, AGORA.Solution))

    return g


def incremental_eval_bgp(ctx, bgp):
    # type: (QueryContext, iter) -> iter
    fragment_generator = ctx.graph.gen(bgp, filters=ctx.filters)
    if fragment_generator is not None:
        dgraph = nx.DiGraph()
        agp = ctx.graph.build_agp(bgp)

        variables = set([v for v in agp.wire.nodes() if isinstance(v, Variable)])
        dgraph.add_nodes_from(variables)

        for c, tp in __base_generator(agp, ctx, fragment_generator):
            [dgraph.add_edge(v, c) for v in c.variables]

            if len(c.variables) == len(variables):
                yield __query_context(ctx, c).solution()
            else:
                if isinstance(tp.o, Variable) and isinstance(tp.s, Variable):
                    for solution in __eval_delta(c, dgraph, c.variables, variables):
                        yield __query_context(ctx, solution).solution()
