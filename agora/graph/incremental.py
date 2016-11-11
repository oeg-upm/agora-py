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

"""
These method recursively evaluate the SPARQL Algebra

evalQuery is the entry-point, it will setup context and
return the SPARQLResult object

evalPart is called on each level and will delegate to the right method

A rdflib.plugins.sparql.sparql.QueryContext is passed along, keeping
information needed for evaluation

A list of dicts (solution mappings) is returned, apart from GroupBy which may
also return a dict of list of dicts

"""

import collections

from rdflib import Variable, Graph, BNode, URIRef, Literal
from rdflib.plugins.sparql import CUSTOM_EVALS
from rdflib.plugins.sparql.aggregates import evalAgg
from rdflib.plugins.sparql.evalutils import (
    _eval, _join, _minus, _fillTemplate, _ebv)
from rdflib.plugins.sparql.parserutils import value
from rdflib.plugins.sparql.sparql import (
    QueryContext, AlreadyBound, FrozenBindings, SPARQLError)


def tp_part(graph, term):
    if isinstance(term, Variable) or isinstance(term, BNode):
        return '?{}'.format(str(term))
    elif isinstance(term, URIRef):
        return '<{}>'.format(term)
    elif isinstance(term, Literal):
        return term.n3(namespace_manager=graph.namespace_manager)


def collect_bgp_fragment(graph, bgp):
    res = graph.gen(bgp)
    if res is not None:
        plan, gen = res
        for c, s, p, o in gen:
            graph.add((s, p, o))


def __bind(ctx, tp, ss, sp, so):
    if ctx[tp.s] is None:
        ctx[tp.s] = ss

    try:
        ctx[tp.p] = sp
    except AlreadyBound:
        pass

    try:
        if ctx[tp.o] is None:
            ctx[tp.o] = so
    except AlreadyBound:
        pass

    return ctx


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


def __base_generator(fragment):
    plan, gen = fragment
    for tp, ss, _, so in gen:
        # print '>>>', ss, _, so
        kwargs = {}
        if isinstance(tp.o, Variable):
            kwargs[tp.o] = so
        if isinstance(tp.s, Variable):
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
    delta = ContextCollection()
    for qc in intermediate:
        if c != qc:
            n = __exploit(c, qc)
            if n is not None and n not in delta and n not in intermediate:
                delta.add(n)
    return __reduce_collection(delta)


def __equal_intersection(c1, c2):
    return all([c1[k] == c2[k] for k in set(c1.map.keys()).intersection(set(c2.map.keys()))])


def __no_containment(c1, c2):
    # type: (Context, Context) -> bool
    return not c1.issubset(c2) and not c2.issubset(c1)


def __part_of(c1, c2):
    return c1.issubset(c2)


def __has_difference(c1, c2):
    return bool(len(set(__diff(c1, c2))))


def __select_candidates(c, col):
    candidates = filter(lambda x: __no_containment(x, c), col)
    candidates = filter(lambda x: __equal_intersection(c, x), candidates)
    candidates = filter(lambda x: __has_difference(c, x), candidates)
    return set(candidates)


def __exploit_collection(col, delta=None, pairs=None):
    if pairs is None:
        pairs = set([])
    if delta is None:
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
                # else:
                #     print '[ x ] discarding duplicate', new
        acum.add(r)

        if r not in candidates and r not in delta:
            delta.add(r)

    return __reduce_collection(delta)


def __joins_with(tp, c, x):
    # type: (TP, Context, Context) -> bool
    any = False
    len_c = len(c.variables)
    if len_c > 1:
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
    for c in col:
        unique = True
        for c2 in col:
            if c != c2 and c.issubset(c2):
                unique = False
                break
        if unique:
            yield c


def __eval_delta(c, tp, intermediate):
    if isinstance(tp.o, Variable) and isinstance(tp.s, Variable):
        candidates = filter(lambda x: __joins_with(tp, c, x), intermediate)
        reduce_candidates = set(__reduce_collection(candidates))
        for r in __exploit_collection(__compose(c, reduce_candidates)):
            yield r

    yield c


def __query_context(ctx, c):
    # type: (QueryContext, Context) -> QueryContext
    q = ctx.push()
    for k, v in c.map.items():
        q[k] = v
    return q


def __incrementalEvalBGP(ctx, gen, wire):
    vars = set([v for v in wire.nodes() if isinstance(v, Variable)])
    contexts = ContextCollection()
    i = 0
    n_triples = 0
    for c, tp in gen:
        for inter in __eval_delta(c, tp, contexts):
            if inter not in contexts:
                if all([inter[k] is not None for k in vars]):
                    yield __query_context(ctx, inter).solution()
                else:
                    contexts.add(inter)
                    # print '[', i, ']', inter
                i += 1
        n_triples += 1
    # print 'total triples=', n_triples


def __evalBGP(ctx, bgp):
    """
    A basic graph pattern
    """

    if not bgp:
        yield ctx.solution()
        return

    s, p, o = bgp[0]

    _s = ctx[s]
    _p = ctx[p]
    _o = ctx[o]

    for ss, sp, so in ctx.graph.triples((_s, _p, _o)):
        if None in (_s, _p, _o):
            c = ctx.push()
        else:
            c = ctx

        if _s is None:
            c[s] = ss

        try:
            if _p is None:
                c[p] = sp
        except AlreadyBound:
            continue

        try:
            if _o is None:
                c[o] = so
        except AlreadyBound:
            continue

        for x in __evalBGP(c, bgp[1:]):
            yield x


def evalBGP(ctx, bgp):
    print 'evaluating BGP {}'.format(bgp)

    fragment_generator = ctx.graph.gen(bgp)
    if fragment_generator is not None:
        agp = ctx.graph.build_agp(bgp)
        for x in __incrementalEvalBGP(ctx, __base_generator(fragment_generator), agp.wire):
            yield x


def evalExtend(ctx, extend):
    # TODO: Deal with dict returned from evalPart from GROUP BY

    print 'evaluating extend {}'.format(extend)

    for c in evalPart(ctx, extend.p):
        try:
            e = _eval(extend.expr, c.forget(ctx))
            if isinstance(e, SPARQLError):
                raise e

            yield c.merge({extend.var: e})

        except SPARQLError:
            yield c


def evalLazyJoin(ctx, join):
    """
    A lazy join will push the variables bound
    in the first part to the second part,
    essentially doing the join implicitly
    hopefully evaluating much fewer triples
    """

    print 'evaluating lazy join {}'.format(join)

    for a in evalPart(ctx, join.p1):
        c = ctx.thaw(a)
        for b in evalPart(c, join.p2):
            yield b


def evalJoin(ctx, join):
    # TODO: Deal with dict returned from evalPart from GROUP BY
    # only ever for join.p1

    print 'evaluating join {}'.format(join)

    if join.lazy:
        return evalLazyJoin(ctx, join)
    else:
        a = evalPart(ctx, join.p1)
        b = set(evalPart(ctx, join.p2))
        return _join(a, b)


def evalUnion(ctx, union):
    res = set()

    for x in evalPart(ctx, union.p1):
        res.add(x)
        yield x
    for x in evalPart(ctx, union.p2):
        if x not in res:
            yield x


def evalMinus(ctx, minus):
    a = evalPart(ctx, minus.p1)
    b = set(evalPart(ctx, minus.p2))
    return _minus(a, b)


def evalLeftJoin(ctx, join):
    # import pdb; pdb.set_trace()

    print 'evaluating left join {}'.format(join)

    for a in evalPart(ctx, join.p1):
        ok = False
        c = ctx.thaw(a)
        for b in evalPart(c, join.p2):
            if _ebv(join.expr, b.forget(ctx)):
                ok = True
                yield b
        if not ok:
            # we've cheated, the ctx above may contain
            # vars bound outside our scope
            # before we yield a solution without the OPTIONAL part
            # check that we would have had no OPTIONAL matches
            # even without prior bindings...
            if not any(_ebv(join.expr, b) for b in
                       evalPart(ctx.thaw(a.remember(join.p1._vars)), join.p2)):
                yield a


def evalFilter(ctx, part):
    # TODO: Deal with dict returned from evalPart!
    for c in evalPart(ctx, part.p):
        if _ebv(part.expr, c.forget(ctx)):
            yield c


def evalGraph(ctx, part):
    if ctx.dataset is None:
        raise Exception(
            "Non-conjunctive-graph doesn't know about " +
            "graphs. Try a query without GRAPH.")

    ctx = ctx.clone()
    graph = ctx[part.term]
    if graph is None:

        for graph in ctx.dataset.contexts():

            # in SPARQL the default graph is NOT a named graph
            if graph == ctx.dataset.default_context:
                continue

            c = ctx.pushGraph(graph)
            c = c.push()
            graphSolution = [{part.term: graph.identifier}]
            for x in _join(evalPart(c, part.p), graphSolution):
                yield x

    else:
        c = ctx.pushGraph(ctx.dataset.get_context(graph))
        for x in evalPart(c, part.p):
            yield x


def evalValues(ctx, part):
    for r in part.p.res:
        c = ctx.push()
        try:
            for k, v in r.iteritems():
                if v != 'UNDEF':
                    c[k] = v
        except AlreadyBound:
            continue

        yield c.solution()


def evalMultiset(ctx, part):
    if part.p.name == 'values':
        return evalValues(ctx, part)

    return evalPart(ctx, part.p)


def evalPart(ctx, part):
    # try custom evaluation functions
    for name, c in CUSTOM_EVALS.items():
        try:
            return c(ctx, part)
        except NotImplementedError:
            pass  # the given custome-function did not handle this part

    if part.name == 'BGP':
        return evalBGP(ctx, part.triples)  # NOTE pass part.triples, not part!
    elif part.name == 'Filter':
        return evalFilter(ctx, part)
    elif part.name == 'Join':
        return evalJoin(ctx, part)
    elif part.name == 'LeftJoin':
        return evalLeftJoin(ctx, part)
    elif part.name == 'Graph':
        return evalGraph(ctx, part)
    elif part.name == 'Union':
        return evalUnion(ctx, part)
    elif part.name == 'ToMultiSet':
        return evalMultiset(ctx, part)
    elif part.name == 'Extend':
        return evalExtend(ctx, part)
    elif part.name == 'Minus':
        return evalMinus(ctx, part)

    elif part.name == 'Project':
        return evalProject(ctx, part)
    elif part.name == 'Slice':
        return evalSlice(ctx, part)
    elif part.name == 'Distinct':
        return evalDistinct(ctx, part)
    elif part.name == 'Reduced':
        return evalReduced(ctx, part)

    elif part.name == 'OrderBy':
        return evalOrderBy(ctx, part)
    elif part.name == 'Group':
        return evalGroup(ctx, part)
    elif part.name == 'AggregateJoin':
        return evalAggregateJoin(ctx, part)

    elif part.name == 'SelectQuery':
        return evalSelectQuery(ctx, part)
    elif part.name == 'AskQuery':
        return evalAskQuery(ctx, part)
    elif part.name == 'ConstructQuery':
        return evalConstructQuery(ctx, part)

    elif part.name == 'ServiceGraphPattern':
        raise Exception('ServiceGraphPattern not implemented')

    elif part.name == 'DescribeQuery':
        raise Exception('DESCRIBE not implemented')

    else:
        # import pdb ; pdb.set_trace()
        raise Exception('I dont know: %s' % part.name)


def evalGroup(ctx, group):
    """
    http://www.w3.org/TR/sparql11-query/#defn_algGroup
    """

    p = evalPart(ctx, group.p)
    if not group.expr:
        return {1: list(p)}
    else:
        res = collections.defaultdict(list)
        for c in p:
            k = tuple(_eval(e, c) for e in group.expr)
            res[k].append(c)
        return res


def evalAggregateJoin(ctx, agg):
    # import pdb ; pdb.set_trace()
    p = evalPart(ctx, agg.p)
    # p is always a Group, we always get a dict back

    for row in p:
        bindings = {}
        for a in agg.A:
            evalAgg(a, p[row], bindings)

        yield FrozenBindings(ctx, bindings)

    if len(p) == 0:
        yield FrozenBindings(ctx)


def evalOrderBy(ctx, part):
    res = evalPart(ctx, part.p)

    for e in reversed(part.expr):

        def val(x):
            v = value(x, e.expr, variables=True)
            if isinstance(v, Variable):
                return (0, v)
            elif isinstance(v, BNode):
                return (1, v)
            elif isinstance(v, URIRef):
                return (2, v)
            elif isinstance(v, Literal):
                return (3, v)

        reverse = bool(e.order and e.order == 'DESC')
        res = sorted(res, key=val, reverse=reverse)

    return res


def evalSlice(ctx, slice):
    # import pdb; pdb.set_trace()
    res = evalPart(ctx, slice.p)
    i = 0
    while i < slice.start:
        res.next()
        i += 1
    i = 0
    for x in res:
        i += 1
        if slice.length is None:
            yield x
        else:
            if i <= slice.length:
                yield x
            else:
                break


def evalReduced(ctx, part):
    return evalPart(ctx, part.p)  # TODO!


def evalDistinct(ctx, part):
    res = evalPart(ctx, part.p)

    done = set()
    for x in res:
        if x not in done:
            yield x
            done.add(x)


def evalProject(ctx, project):
    res = evalPart(ctx, project.p)

    return (row.project(project.PV) for row in res)


def evalSelectQuery(ctx, query):
    res = {}
    res["type_"] = "SELECT"
    res["bindings"] = evalPart(ctx, query.p)
    res["vars_"] = query.PV
    return res


def evalAskQuery(ctx, query):
    res = {}
    res["type_"] = "ASK"
    res["askAnswer"] = False
    for x in evalPart(ctx, query.p):
        res["askAnswer"] = True
        break

    return res


def evalConstructQuery(ctx, query):
    template = query.template

    if not template:
        # a construct-where query
        template = query.p.p.triples  # query->project->bgp ...

    graph = Graph()

    for c in evalPart(ctx, query.p):
        graph += _fillTemplate(template, c)

    res = {}
    res["type_"] = "CONSTRUCT"
    res["graph"] = graph

    return res


def evalQuery(graph, query, initBindings, base=None):
    ctx = QueryContext(graph)

    ctx.prologue = query.prologue

    if initBindings:
        for k, v in initBindings.iteritems():
            if not isinstance(k, Variable):
                k = Variable(k)
            ctx[k] = v
            # ctx.push()  # nescessary?

    main = query.algebra

    # import pdb; pdb.set_trace()
    if main.datasetClause:
        if ctx.dataset is None:
            raise Exception(
                "Non-conjunctive-graph doesn't know about " +
                "graphs! Try a query without FROM (NAMED).")

        ctx = ctx.clone()  # or push/pop?

        firstDefault = False
        for d in main.datasetClause:
            if d.default:

                if firstDefault:
                    # replace current default graph
                    dg = ctx.dataset.get_context(BNode())
                    ctx = ctx.pushGraph(dg)
                    firstDefault = True

                ctx.load(d.default, default=True)

            elif d.named:
                g = d.named
                ctx.load(g, default=False)

    return evalPart(ctx, main)
