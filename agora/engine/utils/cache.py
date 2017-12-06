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
from base64 import b64encode
from functools import wraps

from rdflib import ConjunctiveGraph
from rdflib import Graph

__author__ = 'Fernando Serena'

log = logging.getLogger('agora.engine.utils.cache')


class Cache(dict):
    """
    Special dictionary to be used as a cache that offers mechanisms to watch and clear (on cascade)
    """

    def __init__(self, **kwargs):
        super(Cache, self).__init__(**kwargs)
        self.__watchers = []
        self.stable = 1

    def watch(self, other):
        # type: (Cache) -> None
        if isinstance(other, Cache):
            if self not in other.__watchers:
                other.__watchers.append(self)

    def clear(self):
        # type: () -> None
        super(Cache, self).clear()
        for obs in self.__watchers:
            obs.clear()


def cached(cache, level=0):
    # type: (Cache, int) -> callable
    def d(f):
        @wraps(f)
        def wrap(*args, **kwargs):
            if not isinstance(cache, Cache):
                raise AttributeError('Cache type is not valid')
            cache_key = b64encode(f.__name__ + str(args[1:]) + str(kwargs))
            if not cache.stable >= level or cache_key not in cache:
                result = f(*args, **kwargs)
                cache[cache_key] = result
            return cache[cache_key]

        return wrap

    return d


class SubGraph(Graph):
    def __init__(self, cache, store='default', identifier=None,
                 namespace_manager=None):
        super(SubGraph, self).__init__(store, identifier, namespace_manager)
        self.__cache = cache

    def query(self, q, **kwargs):
        key = b64encode(q + self.identifier)
        if key not in self.__cache:
            r = set(super(SubGraph, self).query(q))
            self.__cache[key] = r
        return self.__cache[key]


class ContextGraph(ConjunctiveGraph):
    def __init__(self, cache, store='default', identifier=None):
        # type: (Cache, str, str) -> ContextGraph
        super(ContextGraph, self).__init__(store, identifier)
        self.__cache = cache

    def query(self, q, **kwargs):
        key = b64encode(q + self.identifier)
        if key not in self.__cache:
            r = set(super(ContextGraph, self).query(q))
            self.__cache[key] = r
        return self.__cache[key]

    def remove_context(self, context):
        super(ContextGraph, self).remove_context(context)
        self.__cache.clear()

    def get_context(self, identifier, quoted=False):
        # type: (any, bool) -> SubGraph
        return SubGraph(self.__cache, store=self.store, identifier=identifier,
                        namespace_manager=self.namespace_manager)
