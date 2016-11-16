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
from rdflib import Variable
from rdflib.plugins.sparql.sparql import AlreadyBound

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
        return self.map == other.map

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
    # type: (AGP, AgoraContext, iter) -> iter
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


def __compose(c, intermediate):
    # type: (Context, iter) -> iter
    delta = ContextCollection()
    for qc in intermediate:
        if c != qc:
            n = __exploit(c, qc)
            if n is not None and n not in delta and n not in intermediate:
                delta.add(n)
    return __reduce_collection(delta)


def __equal_intersection(c1, c2):
    # type: (Context, Context) -> bool
    return all([c1[k] == c2[k] for k in set(c1.map.keys()).intersection(set(c2.map.keys()))])


def __no_containment(c1, c2):
    # type: (Context, Context) -> bool
    return not c1.issubset(c2) and not c2.issubset(c1)


def __part_of(c1, c2):
    # type: (Context, Context) -> bool
    return c1.issubset(c2)


def __has_difference(c1, c2):
    # type: (Context, Context) -> bool
    return bool(len(set(__diff(c1, c2))))


def __select_candidates(c, col):
    # type: (Context, ContextCollection) -> set
    candidates = filter(lambda x: __no_containment(x, c), col)
    candidates = filter(lambda x: __equal_intersection(c, x), candidates)
    candidates = filter(lambda x: __has_difference(c, x), candidates)
    return set(candidates)


def __exploit_collection(col):
    # type: (set, ContextCollection, set) -> iter
    pairs = set([])
    delta = ContextCollection()
    acum = ContextCollection()
    for r in col:
        candidates = set(__select_candidates(r, acum))
        candidates = filter(lambda x: (x, r) not in pairs, candidates)
        for cand in candidates:
            pairs.add((r, cand))
            new = __exploit(r, cand)
            if new is not None:
                if new not in delta:
                    delta.add(new)
        acum.add(r)

        if r not in candidates and r not in delta:
            delta.add(r)

    return __reduce_collection(delta)


def __joins_with(tp, c, x):
    # type: (TP, Context, Context) -> bool
    any = False
    len_c = len(c.variables)
    if len_c > 1 and len(x.variables) > 1:
        if len_c - len(c.variables.intersection(x.variables)) == 1:
            if tp.s in x.variables:
                any = True
                if c[tp.s] != x[tp.s]:
                    return False
            if tp.o in x.variables:
                any = True
                if c[tp.o] != x[tp.o]:
                    return False

    return any


def __reduce_collection(col):
    # type: (iter) -> iter
    for c in col:
        unique = True
        for c2 in col:
            if c != c2 and c.issubset(c2):
                unique = False
                break
        if unique:
            yield c


def __eval_delta(c, tp, intermediate):
    # type: (Context, TP, iter) -> iter
    if isinstance(tp.o, Variable) and isinstance(tp.s, Variable):
        candidates = filter(lambda x: __joins_with(tp, c, x), intermediate)
        reduce_candidates = set(__reduce_collection(candidates))
        compose = set(__compose(c, reduce_candidates))
        exploit = list(__exploit_collection(compose))
        for r in exploit:
            yield r

    yield c


def __query_context(ctx, c):
    # type: (QueryContext, Context) -> QueryContext
    q = ctx.push()
    for k, v in c.map.items():
        try:
            q[k] = v
        except AlreadyBound:
            pass
    return q


def incremental_eval_bgp(ctx, bgp):
    # type: (QueryContext, iter) -> iter
    fragment_generator = ctx.graph.gen(bgp, filters=ctx.filters)
    if fragment_generator is not None:
        agp = ctx.graph.build_agp(bgp)

        variables = set([v for v in agp.wire.nodes() if isinstance(v, Variable)])
        contexts = ContextCollection()
        for c, tp in __base_generator(agp, ctx, fragment_generator):
            for inter in __eval_delta(c, tp, contexts):
                if inter not in contexts:
                    if all([inter[k] is not None for k in variables]):
                        yield __query_context(ctx, inter).solution()
                    else:
                        contexts.add(inter)
