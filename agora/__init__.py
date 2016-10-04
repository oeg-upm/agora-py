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
from agora.graph.fragment import Fragment
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
    def __init__(self):
        self.__fountain = Fountain()
        self.__planner = Planner(self.__fountain)

    @property
    def fountain(self):
        return self.__fountain

    def query(self, query, cache=None, chunk_size=None):
        collector = Collector(self.__planner, cache=cache)
        fragment = Fragment(collector)
        return fragment.query(query, chunk_size=chunk_size)

    def fragment(self, query, cache=None):
        collector = Collector(self.__planner, cache=cache)
        fragment = Fragment(collector)
        agp = fragment.agp(query)
        gen_dict = collector.get_fragment_generator(*agp)
        return gen_dict['generator']

    def graph(self, query, cache=None):
        collector = Collector(self.__planner, cache=cache)
        fragment = Fragment(collector)
        agp = fragment.agp(query)
        gen_dict = collector.get_fragment_generator(*agp)
        graph = Graph()
        for prefix, ns in fragment.namespaces():
            graph.bind(prefix, ns)
        for c, s, p, o in gen_dict['generator']:
            graph.add((s, p, o))
        return graph

    def search_plan(self, query):
        collector = Collector(self.__planner)
        fragment = Fragment(collector)
        return fragment.search_plan(query)

    def agp(self, query):
        collector = Collector(self.__planner)
        fragment = Fragment(collector)
        return fragment.agp(query)
