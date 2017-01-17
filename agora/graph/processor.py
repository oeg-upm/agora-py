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
from datetime import datetime

from agora.graph.evaluate import evalQuery, discriminate_filters
from rdflib.plugins.sparql.algebra import translateQuery
from rdflib.plugins.sparql.parser import parseQuery
from rdflib.plugins.sparql.processor import SPARQLResult, SPARQLProcessor
from rdflib.plugins.sparql.sparql import Query
from rdflib.query import ResultRow

__author__ = 'Fernando Serena'

log = logging.getLogger('agora.graph.processor')


class FragmentResult(SPARQLResult):
    def __init__(self, res):
        super(FragmentResult, self).__init__(res)
        self.chunk_size = res.get('chunk_', None)

    def __iter__(self):
        if self.type in ("CONSTRUCT", "DESCRIBE"):
            for t in self.graph:
                yield t
        elif self.type == 'ASK':
            yield self.askAnswer
        elif self.type == 'SELECT':
            # this iterates over ResultRows of variable bindings
            if self._genbindings:
                for b in self._genbindings:
                    self._bindings.append(b)
                    yield ResultRow(b, self.vars)
                self._genbindings = None
            else:
                for b in self._bindings:
                    yield ResultRow(b, self.vars)


def __traverse_part(part, filters):
    if part.name == 'Filter':
        for v, f in discriminate_filters(part.expr):
            if v not in filters:
                filters[v] = set([])
            filters[v].add(f)
    if part.name == 'BGP':
        yield part
    else:
        if hasattr(part, 'p1') and part.p1 is not None:
            for p in __traverse_part(part.p1, filters):
                yield p
        if hasattr(part, 'p2') and part.p2 is not None:
            for p in __traverse_part(part.p2, filters):
                yield p

    if part.p is not None:
        for p in __traverse_part(part.p, filters):
            yield p


def extract_bgps(query, prefixes):
    parsetree = parseQuery(query)
    query = translateQuery(parsetree, initNs=prefixes)
    part = query.algebra
    filters = {}
    bgps = []

    for p in __traverse_part(part, filters):
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


class FragmentProcessor(SPARQLProcessor):
    def query(
            self, strOrQuery, initBindings={},
            initNs={}, base=None, DEBUG=False, **kwargs):
        """
        Evaluate a query with the given initial bindings, and initial
        namespaces. The given base is used to resolve relative URIs in
        the query and will be overridden by any BASE given in the query.
        """

        start = datetime.now()
        if not isinstance(strOrQuery, Query):
            parsetree = parseQuery(strOrQuery)
            query = translateQuery(parsetree, base, initNs)
        else:
            query = strOrQuery
        try:
            log.debug(
                u'Took {}ms to parse the SPARQL query: {}'.format((datetime.now() - start).total_seconds(),
                                                                  str(strOrQuery)))
        except Exception:
            pass

        eval = evalQuery(self.graph, query, initBindings, base, **kwargs)
        return eval
