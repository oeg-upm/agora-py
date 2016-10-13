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

    def query(self, query, collector=None, cache=None, chunk_size=None, loader=None):
        if collector is None:
            collector = Collector(self._planner, cache=cache)
        collector.loader = loader
        graph = AgoraGraph(collector)
        return graph.query(query, chunk_size=chunk_size)

    def fragment(self, query, collector=None, cache=None, loader=None):
        if collector is None:
            collector = Collector(self._planner, cache=cache)
        collector.loader = loader
        graph = AgoraGraph(collector)
        agp = graph.agp(query)
        return collector.get_fragment_generator(*agp)

    def agp_fragment(self, *agp, **kwargs):
        collector = kwargs.get('collector', None)
        loader = kwargs.get('loader', None)
        cache = kwargs.get('cache', None)
        if collector is None:
            collector = Collector(self._planner, cache=cache)
        collector.loader = loader
        return collector.get_fragment_generator(*agp)

    def graph(self, query, collector=None, cache=None, loader=None):
        if collector is None:
            collector = Collector(self._planner, cache=cache)
        collector.loader = loader
        graph = AgoraGraph(collector)
        agp = graph.agp(query)
        gen_dict = collector.get_fragment_generator(*agp)
        graph = Graph()
        for prefix, ns in graph.namespaces():
            graph.bind(prefix, ns)
        for c, s, p, o in gen_dict['generator']:
            graph.add((s, p, o))
        return graph

    def search_plan(self, query):
        collector = Collector(self._planner)
        graph = AgoraGraph(collector)
        return graph.search_plan(query)

    def agp(self, query):
        collector = Collector(self._planner)
        graph = AgoraGraph(collector)
        return graph.agp(query)
