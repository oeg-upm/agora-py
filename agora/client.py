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
import itertools
from agora.collector.remote import RemoteCollector
from agora.graph import AgoraGraph
from agora.server.fountain import client as fc
from agora.server.planner import client as pc

from agora.server.fragment import client as cc
from rdflib import Graph

__author__ = 'Fernando Serena'


class AgoraClient(object):
    def __init__(self, fountain_host='localhost', fountain_port=5000, planner_host='localhost', planner_port=5000,
                 fragment_host='localhost', fragment_port=5000):
        self._fountain = fc(host=fountain_host, port=fountain_port)
        self._planner = pc(host=planner_host, port=planner_port, fountain=self._fountain)
        self._collector = RemoteCollector(fragment_host, fragment_port, planner=self._planner)
        self._fragment = cc(host=fragment_host, port=fragment_port)

    @property
    def fountain(self):
        return self._fountain

    @property
    def planner(self):
        return self._planner

    def query(self, query, **kwargs):
        graph = AgoraGraph(self._collector)
        return graph.query(query, **kwargs)

    def fragment(self, query=None, agps=None):
        if not (query or agps):
            return

        graph = AgoraGraph(self._collector)
        result = Graph(namespace_manager=graph.namespace_manager)

        if query:
            for c, s, p, o in self._fragment.fragment(query):
                result.add((s, p, o))
        else:
            for agp in agps:
                for c, s, p, o in graph.collector.get_fragment_generator(agp)['generator']:
                    result.add((s, p, o))
        return result

    def fragment_generator(self, query=None, agps=None):
        def comp_gen(gens):
            for gen in [g['generator'] for g in gens]:
                for q in gen:
                    yield q

        if query is not None:
            generator = self._fragment.fragment(query)
            plan = self.search_plan(query)
        else:
            graph = AgoraGraph(self._collector)
            generators = [graph.collector.get_fragment_generator(agp) for agp in agps]
            plan = Graph(namespace_manager=graph.namespace_manager)
            for g in generators:
                plan.__iadd__(g['plan'])
            generator = comp_gen(generators)

        prefixes = dict(plan.namespaces())

        return {'prefixes': prefixes, 'plan': plan, 'generator': generator}

    def search_plan(self, query):
        graph = AgoraGraph(self._collector)
        comp_plan = Graph(namespace_manager=graph.namespace_manager)
        for agp in graph.agps(query):
            comp_plan.__iadd__(self._planner.make_plan(agp))
        return comp_plan

    def agps(self, query):
        graph = AgoraGraph(self._collector)
        return graph.agps(query)
