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

from agora.collector.remote import RemoteCollector
from agora.graph import AgoraGraph
from agora.server.fountain import client as fc
from agora.server.planner import client as pc

__author__ = 'Fernando Serena'


class AgoraClient(object):
    def __init__(self, fountain_host='localhost', fountain_port=5000, planner_host='localhost', planner_port=5000,
                 fragment_host='localhost', fragment_port=5000):
        self._fountain = fc(host=fountain_host, port=fountain_port)
        self._planner = pc(host=planner_host, port=planner_port, fountain=self._fountain)
        self._collector = RemoteCollector(fragment_host, fragment_port, planner=self._planner)

    @property
    def fountain(self):
        return self._fountain

    @property
    def planner(self):
        return self._planner

    def query(self, query, chunk_size=None):
        graph = AgoraGraph(self._collector)
        return graph.query(query, chunk_size=chunk_size)

    def fragment(self, query):
        graph = AgoraGraph(self._collector)
        agp = graph.agp(query)
        return self._collector.get_fragment_generator(*agp)

    def agp_fragment(self, *agp):
        return self._collector.get_fragment_generator(*agp)
