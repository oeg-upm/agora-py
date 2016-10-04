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

import sys
from threading import Thread, Event, Lock

import networkx as nx
from agora.engine.plan.graph import AGORA
from rdflib import BNode
from rdflib import Literal
from rdflib import RDF
from rdflib import URIRef
from rdflib import Variable
from rdflib.plugins.sparql.algebra import translateQuery
from rdflib.plugins.sparql.evaluate import evalQuery
from rdflib.plugins.sparql.parser import parseQuery
from rdflib.plugins.sparql.processor import SPARQLResult, SPARQLProcessor
from rdflib.plugins.sparql.sparql import Query
from rdflib.query import ResultRow

__author__ = 'Fernando Serena'

log = logging.getLogger('agora.graph.processor')


def tp_part(graph, term):
    if isinstance(term, Variable) or isinstance(term, BNode):
        return '?{}'.format(str(term))
    elif isinstance(term, URIRef):
        return '<{}>'.format(term)
    elif isinstance(term, Literal):
        return term.n3(namespace_manager=graph.namespace_manager)


def chunks(l, n=None):
    """
    Yield successive n-sized chunks from l.
    :param l:
    :param n:
    :return: Generated chunks
    """

    if n is None:
        n = sys.maxint

    if getattr(l, '__iter__') is not None:
        l = l.__iter__()
    finished = False
    while not finished:
        chunk = []
        context, s, p, o = l.next()
        last_ctx = context
        chunk.append((s, p, o))
        try:
            while context == last_ctx or len(chunk) < n:
                context, s, p, o = l.next()
                chunk.append((s, p, o))
        except StopIteration:
            finished = True
        yield chunk


class FragmentResult(SPARQLResult):
    def __init__(self, res):
        super(FragmentResult, self).__init__(res)
        self.gen = res['gen_']
        self.restore = res['restore_']
        self.roots = res['roots_']
        self.plan = res['plan_']
        self.agp = res['agp_']
        self.chunk_size = res.get('chunk_', None)
        self.__collecting = None
        self.__ready = False
        self.__event = Event()
        self.__event.clear()
        self.__lock = Lock()

    def __collect(self):
        for chunk in chunks(self.gen, self.chunk_size):
            with self.__lock:
                self.__ready = True
                self.__event.set()
        with self.__lock:
            self.__event.set()
            self.__collecting = False

    def __iter__(self):
        if self.type in ("CONSTRUCT", "DESCRIBE"):
            for t in self.graph:
                yield t
        elif self.type == 'ASK':
            yield self.askAnswer
        elif self.type == 'SELECT':
            # this iterates over ResultRows of variable bindings

            try:
                if self._genbindings:
                    th = Thread(target=self.__collect)
                    self.__collecting = True
                    th.start()
                    while True:
                        self.__event.wait()
                        with self.__lock:
                            self._genbindings = self.restore().get("bindings")
                            self.__event.clear()
                        for b in self._genbindings:
                            if not b:
                                break
                            if b not in self._bindings:
                                self._bindings.append(b)
                                yield ResultRow(b, self.vars)
                        with self.__lock:
                            if not self.__collecting:
                                break
                    th.join()
                    self._genbindings = None
                else:
                    for b in self._bindings:
                        yield ResultRow(b, self.vars)
            except Exception, e:
                log.error(e.message)


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

        if not isinstance(strOrQuery, Query):
            parsetree = parseQuery(strOrQuery)
            query = translateQuery(parsetree, base, initNs)
        else:
            query = strOrQuery

        roots = set([])
        plan = None
        bgp = []
        bgp_graph = nx.DiGraph()
        if fragment_gen is None:
            part = query.algebra
            while part.name != 'BGP':
                part = part.p

            bgp = set([])

            for s, p, o in part.triples:
                s_elm = tp_part(self.graph, s)
                if p == RDF.type:
                    o_elm = self.graph.qname(o)
                    p_elm = 'a'
                else:
                    p_elm = self.graph.qname(p)
                    o_elm = tp_part(self.graph, o)

                bgp.add('{} {} {}'.format(s_elm, p_elm, o_elm))
                bgp_graph.add_edge(s, o, predicate=p)

            # print 'Agora graph pattern:'
            # print '{'
            # for tp in bgp:
            #     print '  ', tp, '.'
            # print '}'
            plan, fragment_gen = self.graph.gen(*bgp)

            roots = filter(lambda x: bgp_graph.in_degree(x) == 0, bgp_graph.nodes())
            for root in roots:
                part.triples = [(root, RDF.type, AGORA.Root)] + part.triples

            # print 'Effective BGP:'
            # print '{'
            # for tp in part.triples:
            #     print '  {} {} {} .'.format(tp[0].n3(), tp[1].n3(), tp[2].n3())
            # print '}'

        eval = evalQuery(self.graph, query, initBindings, base)
        eval['gen_'] = fragment_gen
        eval['restore_'] = self.restore(query, initBindings, base)
        if chunk_size is not None:
            eval['chunk_'] = chunk_size
        eval['roots_'] = roots
        eval['plan_'] = plan
        eval['agp_'] = bgp
        return eval
