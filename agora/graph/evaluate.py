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
from rdflib.plugins.sparql.algebra import translateQuery
from rdflib.plugins.sparql.parser import parseQuery

from agora.graph.incremental import incremental_eval_bgp

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
from rdflib.plugins.sparql.parserutils import value, Expr
from rdflib.plugins.sparql.sparql import (
    QueryContext, AlreadyBound, FrozenBindings, SPARQLError)


def collect_bgp_fragment(ctx, bgp):
    graph = ctx.graph
    gen = graph.gen(bgp, filters=ctx.filters)
    if gen is not None:
        try:
            while gen.next():
                if ctx.stop is not None and ctx.stop.value:
                    break
        except StopIteration:
            pass


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
    # print 'evaluating BGP {}'.format(bgp)
    yielded = []

    if isinstance(ctx, AgoraQueryContext) and ctx.incremental:
        try:
            for x in incremental_eval_bgp(ctx, bgp):
                yielded.append({str(k): str(x[k]) for k in x})
                yield x
        except Exception:
            pass
    else:
        collect_bgp_fragment(ctx, bgp)

    for x in __evalBGP(ctx, bgp):
        if yielded:
            x_str = {str(k): str(x[k]) for k in x}
            if x_str not in yielded:
                yield x
        else:
            yield x


def evalExtend(ctx, extend):
    # TODO: Deal with dict returned from evalPart from GROUP BY

    # print 'evaluating extend {}'.format(extend)

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

    # print 'evaluating lazy join {}'.format(join)

    for a in evalPart(ctx, join.p1):
        c = ctx.thaw(a)
        for b in evalPart(c, join.p2):
            yield b


def evalJoin(ctx, join):
    # TODO: Deal with dict returned from evalPart from GROUP BY
    # only ever for join.p1

    # print 'evaluating join {}'.format(join)

    if join.lazy:
        return evalLazyJoin(ctx, join)
    else:
        a = evalPart(ctx, join.p1)
        b = set(evalPart(ctx, join.p2))
        return _join(a, b)


def evalUnion(ctx, union):
    res = set()

    # print 'evaluating union {}'.format(union)

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


__sparql_op_symbols = {
    'ConditionalAndExpression': '&&',
    'ConditionalOrExpression': '||',
    'UnaryNot': '!',
    'UnaryMinus': '-',
    'UnaryPlus': '+',
    'Function': ','
}


def __serialize_expr(expr, context=None):
    if isinstance(expr, Expr):
        if not expr._vars:
            return expr.eval()
        else:
            if 'Builtin_' in expr.name:
                if 'REGEX' in expr.name:
                    expr_str = u'REGEX({}, "{}"'.format(expr.text.n3(), expr.pattern)
                    if expr.flags:
                        expr_str += u', {}'.format(expr.flags)
                    expr_str += u')'
                    return expr_str
                else:
                    return u'{}({})'.format(expr.name.replace('Builtin_', ''), expr.arg.n3())
            expr_str = __serialize_expr(expr.expr, expr.name)

            if 'Unary' in expr.name:
                return u'{}{}'.format(__sparql_op_symbols[expr.name], expr_str)
            elif 'Function' in expr.name:
                return u'{}({})'.format(expr['iri'].n3(), __serialize_expr(expr.expr, expr.name))

            other_str = __serialize_expr(expr['other'], expr.name)
            if expr.name in __sparql_op_symbols:
                op_str = __sparql_op_symbols[expr.name]
            else:
                op_str = expr['op']
            return u'{} {} {}'.format(expr_str, op_str, other_str)
    elif isinstance(expr, list):
        return __sparql_op_symbols[context].join(map(lambda x: __serialize_expr(x), expr))
    else:
        if isinstance(expr, Variable):
            return expr.n3()

        return expr.n3()


def __serialize_filter(f):
    return u'{}'.format(__serialize_expr(f))


def discriminate_filters(expr):
    if isinstance(expr, Expr):
        n = len(expr._vars)
        if n == 1:
            for v in expr._vars:
                f_str = __serialize_filter(expr)
                yield v, f_str
        elif n > 1 and expr.get('op', None) and expr.name == 'ConditionalAndExpression':
            for f in discriminate_filters(expr.expr):
                yield f
            for e in expr['other']:
                for f in discriminate_filters(e):
                    yield f
    elif isinstance(expr, Variable):
        yield expr, __serialize_filter(expr)


def evalFilter(ctx, part):
    # TODO: Deal with dict returned from evalPart!
    for v, f in discriminate_filters(part.expr):
        if v not in ctx.filters:
            ctx.filters[v] = set([])
        ctx.filters[v].add(f)

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


class AgoraQueryContext(QueryContext):
    def __init__(self, graph=None, bindings=None, incremental=True, stop=None, **kwargs):
        super(AgoraQueryContext, self).__init__(graph, bindings)
        self.incremental = incremental
        self.filters = {}
        self.stop = stop

    def clone(self, bindings=None):
        r = AgoraQueryContext(
            self._dataset if self._dataset is not None else self.graph)
        r.incremental = self.incremental
        r.prologue = self.prologue
        r.bindings.update(bindings or self.bindings)
        r.graph = self.graph
        r.bnodes = self.bnodes
        r.stop = self.stop
        return r


def traverse_part(part, filters):
    if part.name == 'Filter':
        for v, f in discriminate_filters(part.expr):
            if v not in filters:
                filters[v] = set([])
            filters[v].add(f)
    if part.name == 'BGP':
        yield part
    else:
        if hasattr(part, 'p1') and part.p1 is not None:
            for p in traverse_part(part.p1, filters):
                yield p
        if hasattr(part, 'p2') and part.p2 is not None:
            for p in traverse_part(part.p2, filters):
                yield p

    if part.p is not None:
        for p in traverse_part(part.p, filters):
            yield p


def extract_bgps(query, prefixes):
    parsetree = parseQuery(query)
    query = translateQuery(parsetree, initNs=prefixes)
    part = query.algebra
    filters = {}
    bgps = []

    for p in traverse_part(part, filters):
        bgps.append(p)

    for bgp in bgps:
        yield bgp, {v: filters[v] for v in bgp._vars if v in filters}

        # while part:
        #     if part.name == 'Filter':
        #         for v, f in discriminate_filters(part.expr):
        #             if v not in filters:
        #                 filters[v] = set([])
        #             filters[v].add(f)
        #     if part.name == 'BGP':
        #         bgps.append(part)
        #     else:
        #         if hasattr(part, 'p1') and part.p1 is not None:
        #             bgps.append(part.p1)
        #         if hasattr(part, 'p2') and part.p2 is not None:
        #             bgps.append(part.p2)
        #     part = part.p
        #
        # print filters
        #
        # for bgp in bgps:
        #     yield bgp


def evalQuery(graph, query, initBindings, base=None, incremental=True, **kwargs):
    ctx = AgoraQueryContext(graph=graph, incremental=incremental, **kwargs)

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
