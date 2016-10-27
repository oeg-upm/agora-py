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

from agora.graph.evaluate import evalQuery
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


def extract_bgps(query, prefixes):
    parsetree = parseQuery(query)
    query = translateQuery(parsetree, initNs=prefixes)
    part = query.algebra
    while part:
        if part.name == 'BGP':
            yield part
        elif part.name == 'LeftJoin':
            yield part.p1
            yield part.p2
        elif part.name == 'Minus':
            yield part.p1
            yield part.p2
        part = part.p


class FragmentProcessor(SPARQLProcessor):
    def restore(self, query, initBindings, base):
        def wrapper():
            return evalQuery(self.graph, query, initBindings, base)

        return wrapper

    def query(
            self, strOrQuery, initBindings={},
            initNs={}, base=None, DEBUG=False, fragment_gen=None, chunk_size=None, **kwargs):
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
        log.debug(
            'Took {}ms to parse the SPARQL query: {}'.format((datetime.now() - start).total_seconds(), str(strOrQuery)))

        eval = evalQuery(self.graph, query, initBindings, base)
        return eval
