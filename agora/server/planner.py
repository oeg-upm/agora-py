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

from agora.engine.plan import AbstractPlanner
from agora.server import AgoraServer, TURTLE, HTML, AgoraClient
from agora.server.fountain import client as fc
from rdflib import Graph

__author__ = 'Fernando Serena'

log = logging.getLogger('agora.server.planner')


def build(planner, server=None, import_name=__name__):
    # type: (AbstractPlanner, AgoraServer, str) -> AgoraServer

    if server is None:
        server = AgoraServer(import_name)

    @server.get('/plan', produce_types=(TURTLE, HTML))
    def make_plan():
        gp_str = server.request_args.get('gp', '{}')
        gp_str = gp_str.lstrip('{').rstrip('}').strip()
        tps = re.split('\. ', gp_str)
        tps = map(lambda x: x.strip(), filter(lambda y: y != '', tps))
        plan = planner.make_plan(*tps)
        return plan.serialize(format='turtle')

    return server


class PlannerClient(AgoraClient, AbstractPlanner):
    def __init__(self, host='localhost', port=9002):
        super(PlannerClient, self).__init__(host, port)
        self.__fountain = fc(host, port)

    def make_plan(self, *tps):
        agp = '{ %s }' % ' . '.join(tps)
        response = self._get_request('plan?gp=%s' % agp, accept='text/turtle')
        graph = Graph()
        graph.parse(StringIO(response), format='text/turtle')
        return graph

    @property
    def fountain(self):
        return self.__fountain


def client(host='localhost', port=9002):
    # type: (str, int) -> PlannerClient
    return PlannerClient(host, port)
