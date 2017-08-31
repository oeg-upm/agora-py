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
from threading import RLock

from rdflib import ConjunctiveGraph

from agora.engine.utils import prepare_store_path
from agora.engine.utils.cache import ContextGraph

__author__ = 'Fernando Serena'

log = logging.getLogger('agora.engine.utils.cache')


def get_cached_triple_store(cache, persist_mode=False, base='store', path='', **kwargs):
    if persist_mode:
        prepare_store_path(base, path)
        graph = ContextGraph(cache, 'Sleepycat')
        graph.open('{}/{}'.format(base, path), create=True)
    else:
        graph = ContextGraph(cache)

    graph.store.graph_aware = False
    return graph


_stores = {}
_st_lock = RLock()


def get_triple_store(persist_mode=False, base='store', path='', **kwargs):
    if persist_mode:
        prepare_store_path(base, path)
        graph = ConjunctiveGraph('Sleepycat', identifier=path)
        graph.open('{}/{}'.format(base, path), create=True)
    else:
        graph = ConjunctiveGraph()
        graph.store.graph_aware = False

    return graph


def rmtree(path):
    try:
        shutil.rmtree(path)
    except OSError:
        pass
