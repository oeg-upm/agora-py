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

from agora.engine.fountain.seed import SeedManager

from agora.engine.fountain.path import PathManager

from agora.engine.utils.kv import get_kv, close

from agora.engine.utils.graph import get_cached_triple_store

from agora.engine.fountain.index import Index

from agora.engine.fountain.schema import Schema

from agora.collector import Collector
from agora.collector.cache import RedisCache
from agora.engine.fountain import Fountain
from agora.engine.plan import Planner, AbstractPlanner
from agora.graph import AgoraGraph
from agora.server.fountain import FountainClient
from agora.server.fountain import client as fc
from agora.server.planner import client as pc
from agora.engine.utils import stopped
from rdflib import Graph
import signal
import sys

__author__ = 'Fernando Serena'


def signal_term_handler(signal, frame):
    print 'got SIGTERM'
    Agora.close()
    # sys.exit(0)


signal.signal(signal.SIGTERM, signal_term_handler)


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
        pass

    @property
    def fountain(self):
        return self.planner.fountain

    @property
    def planner(self):
        # type: () -> AbstractPlanner
        return self._planner

    @planner.setter
    def planner(self, p):
        self._planner = p

    def __get_agora_graph(self, collector, cache, loader, force_seed):
        if collector is None:
            collector = Collector()
            collector.planner = self.planner
            collector.cache = cache
        if loader is not None:
            collector.loader = loader
        if force_seed is not None:
            collector.force_seed = force_seed
        return AgoraGraph(collector)

    def query(self, query, collector=None, cache=None, loader=None, force_seed=None, **kwargs):
        graph = self.__get_agora_graph(collector, cache, loader, force_seed)
        return graph.query(query, collector=collector, **kwargs)

    def fragment(self, query=None, agps=None, collector=None, cache=None, loader=None, force_seed=None):
        if not (query or agps):
            return

        graph = self.__get_agora_graph(collector, cache, loader, force_seed)
        result = Graph(namespace_manager=graph.namespace_manager)

        agps = list(graph.agps(query)) if query else agps

        for agp in agps:
            for c, s, p, o in graph.collector.get_fragment_generator(agp)['generator']:
                result.add((s, p, o))
        return result

    def fragment_generator(self, query=None, agps=None, collector=None, cache=None, loader=None, force_seed=None):
        def comp_gen(gens):
            for gen in [g['generator'] for g in gens]:
                for q in gen:
                    yield q

        graph = self.__get_agora_graph(collector, cache, loader, force_seed)
        agps = list(graph.agps(query)) if query else agps

        generators = [graph.collector.get_fragment_generator(agp, filters=filters) for agp, filters in agps]
        prefixes = {}
        comp_plan = Graph(namespace_manager=graph.namespace_manager)
        for g in generators:
            comp_plan.__iadd__(g['plan'])
            prefixes.update(g['prefixes'])

        return {'prefixes': prefixes, 'plan': comp_plan, 'generator': comp_gen(generators), 'gens': generators}

    def search_plan(self, query, force_seed=None):
        collector = Collector()
        collector.planner = self.planner
        graph = AgoraGraph(collector)
        comp_plan = Graph(namespace_manager=graph.namespace_manager)
        for agp, filters in graph.agps(query):
            comp_plan.__iadd__(self._planner.make_plan(agp, force_seed))
        return comp_plan

    def agp(self, query):
        collector = Collector()
        collector.planner = self.planner
        graph = AgoraGraph(collector)
        return graph.agps(query)

    def __new__(cls, **kwargs):
        a = super(Agora, cls).__new__(cls)
        kv = get_kv(**kwargs)
        schema = Schema()
        schema.graph = get_cached_triple_store(schema.cache, **kwargs)
        index = Index()
        index.r = kv
        index.schema = schema
        sm = SeedManager()
        sm.index = index
        pm = PathManager()
        pm.index = index
        pm.seed_manager = sm

        fountain = Fountain()
        fountain.schema = schema
        fountain.index = index
        fountain.seed_manager = sm
        fountain.path_manager = pm
        planner = Planner(fountain)
        a.planner = planner

        return a

    @staticmethod
    def close():
        close()
        stopped.set()
