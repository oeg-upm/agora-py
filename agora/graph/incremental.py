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
from Queue import Queue, Empty
from threading import Thread

import networkx as nx
from rdflib import Variable
from rdflib.plugins.sparql.sparql import AlreadyBound
from rdflib.plugins.sparql.sparql import QueryContext

from agora.engine.plan import AGP

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
            return self.map.items() == other.map.items()
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
        if ctx.stop is not None:
            if ctx.stop.value > 0:
                break

        if str(tp) in ignore_tps:
            continue
        kwargs = {}
        if isinstance(tp.o, Variable):
            kwargs[tp.o] = so
        if isinstance(tp.s, Variable):
            kwargs[tp.s] = ss

        if kwargs:
            yield Context(**kwargs), tp


def __exploit(c1, c2):
    # type: (Context, Context) -> Context
    new_dict = {v: c1[v] for v in c1.map}
    for v in set.difference(set(c2.map.keys()), set(c1.map.keys())):
        new_dict[v] = c2[v]

    c = Context(**new_dict)
    return c


def __joins(c, x):
    # type: (Context, Context) -> bool
    intersection = c.variables.intersection(x.variables)
    for v in intersection:
        if c[v] != x[v]:
            return False

    return bool(intersection)


def common_descendants(graph, x, c, base):
    if base:
        return False
    try:
        for dx in nx.descendants(graph, c):
            for dc in nx.descendants(graph, x):
                if dx.map == dc.map:
                    return True
                    # if graph.out_degree(dx) > 0:
                    #     return True
    except Exception as e:
        # print e.message
        raise e
    return False


def filter_successor(graph, c, x, base):
    return c.map != x.map and __joins(c, x) and not common_descendants(graph, c, x, base)


def union(x, y):
    r = ContextCollection()
    for c in x:
        r.add(c)
    for c in y:
        r.add(c)
    return r


def __eval_delta(c, graph, roots, variables, base=False):
    # type: (Context, nx.DiGraph) -> iter

    solutions = ContextCollection()

    root_candidates = reduce(lambda x, y: union(x, set(graph.successors(y))), roots, set())
    root_candidates = filter(lambda x: set(x.variables).symmetric_difference(c.variables), root_candidates)
    for root in root_candidates:
        if filter_successor(graph, c, root, base):
            inter = __exploit(root, c)

            if inter not in graph:
                graph.add_node(inter)
                graph.add_edge(root, inter)
                graph.add_edge(c, inter)
            else:
                continue

            if len(graph.nodes()) > 1000:
                break

            # print 'DELTA ({}) {}'.format(len(inter.variables), inter)
            if len(inter.variables) == len(variables):
                solutions.add(inter)
            else:
                pred = filter(lambda x: graph.out_degree(x) > 1, [root, c])
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


def __generate(data):
    queue, agp, ctx, generator = data['queue'], data['agp'], data['context'], data['gen']
    for c, tp in __base_generator(agp, ctx, generator):
        queue.put((c, tp))

    data['collecting'] = False


def incremental_eval_bgp(ctx, bgp):
    # type: (QueryContext, iter) -> iter
    fragment_generator = ctx.graph.gen(bgp, filters=ctx.filters)
    queue = Queue()
    if fragment_generator is not None:
        dgraph = nx.DiGraph()
        agp = ctx.graph.build_agp(bgp)

        variables = set([v for v in agp.wire.nodes() if isinstance(v, Variable)])
        dgraph.add_nodes_from(variables)

        gen_data = {
            'queue': queue,
            'agp': agp,
            'context': ctx,
            'gen': fragment_generator,
            'collecting': True
        }

        gen_thread = Thread(target=__generate, args=(gen_data,))
        gen_thread.start()

        try:
            while gen_data['collecting'] or not queue.empty():
                if ctx.stop is not None:
                    if ctx.stop.value > 0:
                        raise StopIteration()
                try:
                    c, tp = queue.get(timeout=1.0)
                    if len(dgraph.nodes()) > 1000:
                        raise Exception()
                    [dgraph.add_edge(v, c) for v in c.variables]

                    # print 'BASE ({}) {}'.format(len(c.variables), c)
                    if len(c.variables) == len(variables):
                        yield __query_context(ctx, c).solution()
                    else:
                        if isinstance(tp.o, Variable) and isinstance(tp.s, Variable):
                            for solution in __eval_delta(c, dgraph, c.variables, variables, base=True):
                                yield __query_context(ctx, solution).solution()
                except Empty:
                    pass

            # for v in variables:
            #     variable_succ = dgraph.successors(v)
            #     for c in variable_succ:
            #         for solution in __eval_delta(c, dgraph, c.variables, variables, base=True):
            #             yield __query_context(ctx, solution).solution()
        finally:
            gen_thread.join()


