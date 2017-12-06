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
import re
from StringIO import StringIO

from rdflib import Graph, URIRef, BNode

from agora.engine.fountain import AbstractFountain
from agora.engine.plan import AGP
from agora.engine.plan import AbstractPlanner
from agora.server import Server, TURTLE, HTML, Client, APIError, JSON, tuples_force_seed, dict_force_seed
from agora.server.fountain import client as fc

__author__ = 'Fernando Serena'

log = logging.getLogger('agora.server.planner')


def skolemize(g):
    def _skolem(n):
        if isinstance(n, BNode):
            if n not in bnodes_map:
                bnodes_map[n] = URIRef('http://bnodes/{}'.format(str(n)))
            n = bnodes_map[n]
        return n

    bnodes_map = {}
    skolem = Graph()
    for s, p, o in g:
        s = _skolem(s)
        o = _skolem(o)
        skolem.add((s, p, o))
    return skolem


def deskolemize(g):
    def _deskolem(n):
        if isinstance(n, URIRef) and n.startswith('http://bnodes'):
            if n not in bnodes_map:
                bnodes_map[n] = BNode(n.replace('http://bnodes/', ''))
            n = bnodes_map[n]
        return n

    bnodes_map = {}
    deskolem = Graph()
    for s, p, o in g:
        s = _deskolem(s)
        o = _deskolem(o)
        deskolem.add((s, p, o))
    return deskolem


def build(planner, server=None, import_name=__name__):
    # type: (AbstractPlanner, Server, str) -> Server

    if server is None:
        server = Server(import_name)

    def __make_plan(agp_str, force_seed):
        try:
            agp_str = agp_str.lstrip('{').rstrip('}').strip()
            tps = re.split('\. ', agp_str)
            tps = map(lambda x: x.rstrip('.').strip(), filter(lambda y: y != '', tps))
            agp = AGP(tps, prefixes=planner.fountain.prefixes)
            plan = planner.make_plan(agp, force_seed=force_seed)
            plan = skolemize(plan)
            return plan.serialize(format='turtle')
        except NameError, e:
            raise APIError(e.message)

    @server.get('/plan', produce_types=(TURTLE, HTML))
    def make_plan():
        # type: () -> str
        agp_str = server.request_args.get('agp', '{}')
        return __make_plan(agp_str, None)

    @server.post('/plan', produce_types=(TURTLE, HTML), consume_types=(JSON,))
    def make_plan_force_seeds(req_json):
        # type: () -> str
        force_seed = list(tuples_force_seed(req_json['force_seed']))
        return __make_plan(req_json['agp'], force_seed)

    return server


class PlannerClient(Client, AbstractPlanner):
    def __init__(self, host='localhost', port=5000, fountain=None):
        # type: (str, int, AbstractFountain) -> PlannerClient
        super(PlannerClient, self).__init__(host, port)
        self.__fountain = fc(host, port) if fountain is None else fountain

    def make_plan(self, agp, force_seed=None):
        # type: (AGP) -> Graph
        if force_seed:
            url = 'plan'
            type_seed_dict = dict_force_seed(force_seed)
            req_json = {'agp': str(agp), 'force_seed': type_seed_dict}
            response = self._post_request(url, req_json, content_type='application/json', accept='text/turtle')
        else:
            url = 'plan?agp=%s' % agp
            response = self._get_request(url, accept='text/turtle')

        graph = Graph()
        graph.parse(StringIO(response), format='text/turtle')
        graph = deskolemize(graph)

        return graph

    @property
    def fountain(self):
        # type: () -> AbstractFountain
        return self.__fountain


def client(host='localhost', port=5000, fountain=None):
    # type: (str, int) -> PlannerClient
    return PlannerClient(host, port, fountain)
