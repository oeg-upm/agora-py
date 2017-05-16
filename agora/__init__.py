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

from agora.collector import Collector
from agora.collector.cache import RedisCache
from agora.engine.fountain import Fountain
from agora.engine.plan import Planner
from agora.graph import AgoraGraph
from agora.server.fountain import FountainClient
from agora.server.fountain import client as fc
from agora.server.planner import client as pc
from rdflib import Graph

__author__ = 'Fernando Serena'


def setup_logging(level):
    log = logging.getLogger('agora')
    log.setLevel(level)
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setLevel(level)
    ch.setFormatter(formatter)
    log.addHandler(ch)


class Agora(object):
    def __init__(self, **kwargs):
        self._fountain = Fountain(**kwargs)
        self._planner = Planner(self._fountain)

    @property
    def fountain(self):
        return self._fountain

    @property
    def planner(self):
        return self._planner

    def __get_agora_graph(self, collector, cache, loader):
        collector = collector if collector is not None else Collector(self._planner, cache=cache)
        if loader is not None:
            collector.loader = loader
        return AgoraGraph(collector)

    def query(self, query, collector=None, cache=None, loader=None, **kwargs):
        graph = self.__get_agora_graph(collector, cache, loader)
        return graph.query(query, collector=collector, **kwargs)

    def fragment(self, query=None, agps=None, collector=None, cache=None, loader=None):
        if not (query or agps):
            return

        graph = self.__get_agora_graph(collector, cache, loader)
        result = Graph(namespace_manager=graph.namespace_manager)

        agps = list(graph.agps(query)) if query else agps

        for agp in agps:
            for c, s, p, o in graph.collector.get_fragment_generator(agp)['generator']:
                result.add((s, p, o))
        return result

    def fragment_generator(self, query=None, agps=None, collector=None, cache=None, loader=None):
        def comp_gen(gens):
            for gen in [g['generator'] for g in gens]:
                for q in gen:
                    yield q

        graph = self.__get_agora_graph(collector, cache, loader)
        agps = list(graph.agps(query)) if query else agps

        generators = [graph.collector.get_fragment_generator(agp, filters=filters) for agp, filters in agps]
        prefixes = {}
        comp_plan = Graph(namespace_manager=graph.namespace_manager)
        for g in generators:
            comp_plan.__iadd__(g['plan'])
            prefixes.update(g['prefixes'])

        return {'prefixes': prefixes, 'plan': comp_plan, 'generator': comp_gen(generators), 'gens': generators}

    def search_plan(self, query):
        collector = Collector(self._planner)
        graph = AgoraGraph(collector)
        comp_plan = Graph(namespace_manager=graph.namespace_manager)
        for agp, filters in graph.agps(query):
            comp_plan.__iadd__(self._planner.make_plan(agp))
        return comp_plan

    def agp(self, query):
        collector = Collector(self._planner)
        graph = AgoraGraph(collector)
        return graph.agps(query)
