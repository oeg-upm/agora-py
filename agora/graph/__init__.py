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

import rdflib
from agora.engine.plan.graph import AGORA
from rdflib import Graph
from rdflib import RDF
from rdflib import RDFS
from rdflib import URIRef
from rdflib import plugin

__author__ = 'Fernando Serena'

log = logging.getLogger('agora.graph.processor')

plugin.register('agora', rdflib.query.Processor, 'agora.graph.processor', 'FragmentProcessor')
plugin.register('agora', rdflib.query.Result, 'agora.graph.processor', 'FragmentResult')


def extract_tp_from_plan(plan):
    # type: (Graph) -> dict
    """

    :param plan: Search Plan graph
    :return: A string triple representing the pattern for all triple pattern nodes
    """

    def extract_node_id(node):
        nid = node
        if (node, RDF.type, AGORA.Variable) in plan:
            nid = list(plan.objects(node, RDFS.label)).pop()
        elif (node, RDF.type, AGORA.Literal) in plan:
            nid = list(plan.objects(node, AGORA.value)).pop()
        return nid

    def process_tp_node(tpn):
        predicate = list(plan.objects(tpn, AGORA.predicate)).pop()
        subject_node = list(plan.objects(tpn, AGORA.subject)).pop()
        object_node = list(plan.objects(tpn, AGORA.object)).pop()
        subject = extract_node_id(subject_node)
        obj = extract_node_id(object_node)

        return str(subject), predicate.n3(plan.namespace_manager), str(obj)

    return {str(list(plan.objects(tpn, RDFS.label)).pop()): process_tp_node(tpn) for tpn in
            plan.subjects(RDF.type, AGORA.TriplePattern)}


class AgoraGraph(Graph):
    def __init__(self, collector):
        super(AgoraGraph, self).__init__()
        self.__collector = collector
        self.__plan = None
        self.__roots = set([])
        for prefix, ns in collector.prefixes.items():
            self.bind(prefix, ns)

    def gen(self, *tps):
        # type: (list) -> (Graph, iter)
        gen_dict = self.__collector.get_fragment_generator(*tps)
        self.__plan = gen_dict.get('plan', None)
        tps = extract_tp_from_plan(self.__plan) if self.__plan is not None else None
        return self.__plan, self._produce(gen_dict['generator'], tps=tps)

    def _produce(self, gen, tps=None):
        self.remove((None, None, None))
        for context, s, p, o in gen:
            log.debug('Got triple: {} {} {} .'.format(s.encode('utf8', 'replace'), p.encode('utf8', 'replace'),
                                                     o.encode('utf8', 'replace')))
            self.add((s, p, o))
            subject = context[0] if tps is None else tps[str(context)][0]
            if subject in self.__roots:
                self.add((s, RDF.type, AGORA.Root))
            yield context, s, p, o

    def load(self, *tps):
        for x in self.gen(*tps):
            print x

    def query(self, query_object, **kwargs):
        result = super(AgoraGraph, self).query(query_object, processor='agora',
                                               result='agora', **kwargs)
        for root in result.roots:
            root_str = '?{}'.format(root) if not isinstance(root, URIRef) else str(root)
            self.__roots.add(root_str)
        return result

    def search_plan(self, query_object, **kwargs):
        result = super(AgoraGraph, self).query(query_object, processor='agora',
                                               result='agora', **kwargs)

        return result.plan

    def agp(self, query_object, **kwargs):
        result = super(AgoraGraph, self).query(query_object, processor='agora',
                                               result='agora', **kwargs)

        return result.agp
