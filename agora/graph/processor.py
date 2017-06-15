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

from rdflib.plugins.sparql.algebra import translateQuery
from rdflib.plugins.sparql.parser import parseQuery
from rdflib.plugins.sparql.processor import SPARQLResult, SPARQLProcessor
from rdflib.plugins.sparql.sparql import Query
from rdflib.query import ResultRow

from agora.graph.evaluate import evalQuery, traverse_part

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


class FragmentProcessor(SPARQLProcessor):
    def query(
            self, strOrQuery, initBindings={},
            initNs={}, base=None, DEBUG=False, **kwargs):
        """
        Evaluate a query with the given initial bindings, and initial
        namespaces. The given base is used to resolve relative URIs in
        the query and will be overridden by any BASE given in the query.
        """

        if not isinstance(strOrQuery, Query):
            parsetree = parseQuery(strOrQuery)
            query = translateQuery(parsetree, base, initNs)
        else:
            query = strOrQuery

        collector = kwargs.get('collector', None)
        graph = kwargs.get('graph', None)
        if collector is not None and graph is not None:
            part = query.algebra
            filters = {}
            bgps = []

            for p in traverse_part(part, filters):
                bgps.append(p)

            incremental = False
            for bgp in bgps:
                filters = {v: filters[v] for v in bgp._vars if v in filters}
                f = collector.get(graph.build_agp(bgp.triples), filters)
                if f is None or not f['fragment'].updated:
                    incremental = True
                    break
            kwargs['incremental'] = incremental

        del kwargs['collector']
        del kwargs['graph']
        eval = evalQuery(self.graph, query, initBindings, base, **kwargs)
        return eval
