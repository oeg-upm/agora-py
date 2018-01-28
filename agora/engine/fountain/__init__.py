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
import hashlib
import logging
from abc import abstractmethod
from multiprocessing import Lock

from rdflib import URIRef

from agora.engine.fountain import onto as manager
from agora.engine.fountain.index import Index
from agora.engine.fountain.path import PathManager
from agora.engine.fountain.schema import Schema
from agora.engine.fountain.seed import SeedManager

__author__ = 'Fernando Serena'

log = logging.getLogger('agora.engine.fountain')


class FountainError(Exception):
    pass


class AbstractFountain(object):
    @abstractmethod
    def add_vocabulary(self, owl):
        # type: (str) -> iter
        raise NotImplementedError

    @abstractmethod
    def update_vocabulary(self, vid, owl):
        # type: (str, str) -> None
        raise NotImplementedError

    @abstractmethod
    def delete_vocabulary(self, vid):
        # type: (str) -> None
        raise NotImplementedError

    @abstractmethod
    def get_vocabulary(self, vid):
        # type: (str) -> str
        raise NotImplementedError

    @property
    @abstractmethod
    def vocabularies(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def types(self):
        # type: () -> iter
        raise NotImplementedError

    @property
    @abstractmethod
    def properties(self):
        # type: () -> iter
        raise NotImplementedError

    @abstractmethod
    def get_type(self, type):
        raise NotImplementedError

    @abstractmethod
    def get_property(self, property):
        raise NotImplementedError

    @abstractmethod
    def get_paths(self, elm, force_seed=None):
        # type: (str, iter) -> (iter, iter)
        raise NotImplementedError

    @abstractmethod
    def add_seed(self, uri, type):
        # type: (str, str) -> str
        raise NotImplementedError

    @property
    @abstractmethod
    def prefixes(self):
        raise NotImplementedError

    @abstractmethod
    def add_prefixes(self, prefixes):
        raise NotImplementedError

    @property
    @abstractmethod
    def seeds(self):
        # type: () -> iter
        raise NotImplementedError

    @abstractmethod
    def get_seed(self, sid):
        # type: (str) -> dict
        raise NotImplementedError

    @abstractmethod
    def get_type_seeds(self, type):
        # type: (str) -> iter
        raise NotImplementedError

    @abstractmethod
    def delete_seed(self, sid):
        # type: (str) -> None
        raise NotImplementedError

    @abstractmethod
    def get_seed_type_digest(self, type):
        # type: (str) -> None
        raise NotImplementedError

    @abstractmethod
    def delete_type_seeds(self, type):
        # type: (str) -> None
        raise NotImplementedError

    @abstractmethod
    def connected(self, source, target):
        # type: (str, str) -> bool
        raise NotImplementedError


class Fountain(AbstractFountain):
    def __init__(self):
        self.__index = None
        self.__schema = None
        self.__pm = None
        self.__sm = None
        self.__lock = Lock()

    @property
    def index(self):
        # type: () -> Index
        return self.__index

    @index.setter
    def index(self, i):
        # type: (Index) -> None
        self.__index = i
        if self.__schema:
            self.__index.schema = self.__schema

    @property
    def seed_manager(self):
        # type: () -> SeedManager
        return self.__sm

    @seed_manager.setter
    def seed_manager(self, s):
        # type: (SeedManager) -> None
        self.__sm = s

    @property
    def schema(self):
        # type: () -> Schema
        return self.__schema

    @schema.setter
    def schema(self, s):
        # type: (Schema) -> None
        self.__schema = s
        if self.__index:
            self.__index.schema = self.__schema

    @property
    def path_manager(self):
        # type: () -> PathManager
        return self.__pm

    @path_manager.setter
    def path_manager(self, p):
        # type: (PathManager) -> None
        self.__pm = p

    def add_vocabulary(self, owl):
        # type: (str) -> iter
        with self.__lock:
            added_vocs_iter = manager.add_vocabulary(self.__schema, owl)
            self.__schema.cache.stable = 0
            for vid in reversed(added_vocs_iter):
                self.__index.index_vocabulary(vid)
            self.__pm.calculate()
            self.__sm.validate()
            self.__schema.cache.stable = 1
            return added_vocs_iter

    def update_vocabulary(self, vid, owl):
        # type: (str, str) -> None
        with self.__lock:
            manager.update_vocabulary(self.__schema, vid, owl)
            self.__schema.cache.stable = 0
            for v in manager.get_vocabularies(self.__schema):
                self.__index.index_vocabulary(v)
            self.__pm.calculate()
            self.__sm.validate()
            self.__schema.cache.stable = 1

    def delete_vocabulary(self, vid):
        # type: (str) -> None
        with self.__lock:
            manager.delete_vocabulary(self.__schema, vid)
            self.__schema.cache.stable = 0
            self.__index.delete_vocabulary(vid)
            for v in manager.get_vocabularies(self.__schema):
                self.__index.index_vocabulary(v)
            self.__pm.calculate()
            self.__sm.validate()
            self.__schema.cache.stable = 1

    def get_vocabulary(self, vid):
        # type: (str) -> str
        with self.__lock:
            return manager.get_vocabulary(self.__schema, vid)

    @property
    def vocabularies(self):
        return self.__schema.contexts

    @property
    def types(self):
        # type: () -> iter
        return self.__index.types

    @property
    def properties(self):
        # type: () -> iter
        return self.__index.properties

    def get_type(self, type):
        return self.__index.get_type(type)

    def get_property(self, property):
        return self.__index.get_property(property)

    def get_paths(self, elm, force_seed=None):
        # type: (str, iter) -> (iter, iter)
        return self.__pm.get_paths(elm, force_seed=force_seed)

    def connected(self, source, target):
        # type: (str, str) -> bool
        return self.__pm.are_connected(source, target)

    def add_seed(self, uri, type):
        # type: (str, str) -> str
        return self.__sm.add_seed(uri, type)

    @property
    def prefixes(self):
        return self.__index.schema.prefixes

    def add_prefixes(self, prefixes):
        current_ns = dict(self.__schema.graph.namespaces()).values()
        for prefix, ns in prefixes.items():
            if not (prefix.startswith('ns') and ns in current_ns):
                self.__schema.graph.namespace_manager.bind(prefix, URIRef(ns), replace=True, override=True)
        self.__schema.update_ns_dicts()
        self.__schema.cache.stable = 0
        for vid in self.vocabularies:
            self.__index.index_vocabulary(vid)
        self.__pm.calculate()
        self.__sm.validate()
        self.__schema.cache.stable = 1

    @property
    def seeds(self):
        # type: () -> iter
        return self.__sm.seeds

    def get_seed(self, sid):
        # type: (str) -> dict
        return self.__sm.get_seed(sid)

    def get_type_seeds(self, type):
        # type: (str) -> iter
        return self.__sm.get_type_seeds(type)

    def delete_seed(self, sid):
        self.__sm.delete_seed(sid)

    def delete_type_seeds(self, type):
        return self.__sm.delete_type_seeds(type)

    def get_seed_type_digest(self, type):
        m = hashlib.md5()
        for seed in sorted(self.__sm.get_type_seeds(type)):
            m.update(seed)
        return m.digest().encode('base64').strip()
