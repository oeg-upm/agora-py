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
import shutil

import os
from agora.engine.utils.cache import ContextGraph
from rdflib import ConjunctiveGraph

__author__ = 'Fernando Serena'

log = logging.getLogger('agora.engine.utils.cache')


def get_triple_store(cache, persist_mode=False, triples_path=None):
    if persist_mode:
        if triples_path is None:
            triples_path = 'triples'
        graph = ContextGraph(cache, 'Sleepycat')
        graph.open(triples_path, create=True)
    else:
        graph = ContextGraph(cache)

    graph.store.graph_aware = False
    return graph


def get_resource_cache(persist_mode=False, base='store', path=None):
    if persist_mode:
        if path is None:
            path = 'resources'
        if not os.path.exists(base):
            os.makedirs(base)
        if os.path.exists('{}/{}'.format(base, path)):
            shutil.rmtree('{}/{}'.format(base, path))
        os.makedirs('{}/{}'.format(base, path))
        resources_cache = ConjunctiveGraph('Sleepycat')
        resources_cache.open('{}/{}'.format(base, path), create=True)
    else:
        resources_cache = ConjunctiveGraph()
    return resources_cache


def rmtree(path):
    try:
        shutil.rmtree(path)
    except OSError:
        pass
