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

from abc import abstractmethod

import redis
from agora.engine.fountain import onto as manager
from agora.engine.fountain.index import Index
from agora.engine.fountain.path import PathManager
from agora.engine.fountain.schema import Schema
from agora.engine.fountain.seed import SeedManager
from agora.engine.utils.kv import get_kv

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
    def get_paths(self, elm):
        # type: (str) -> (iter, iter)
        raise NotImplementedError

    @abstractmethod
    def add_seed(self, uri, type):
        # type: (str, str) -> str
        raise NotImplementedError

    @property
    @abstractmethod
    def prefixes(self):
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
    def delete_type_seeds(self, type):
        # type: (str) -> None
        raise NotImplementedError


class Fountain(AbstractFountain):
    def __init__(self):

        self.__schema = Schema()
        self.__index = Index()
        self.__index.schema = self.__schema
        self.__sm = SeedManager(self.__index)
        self.__pm = PathManager(self.__index, self.__sm)

    @property
    def index(self):
        return self.__index

    @property
    def seed_manager(self):
        return self.__sm

    @property
    def schema(self):
        return self.__schema

    @property
    def path_manager(self):
        return self.__pm

    def add_vocabulary(self, owl):
        # type: (str) -> iter
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
        manager.update_vocabulary(self.__schema, vid, owl)
        self.__pm.calculate()
        self.__sm.validate()

    def delete_vocabulary(self, vid):
        # type: (str) -> None
        manager.delete_vocabulary(self.__schema, vid)
        self.__pm.calculate()
        self.__sm.validate()

    def get_vocabulary(self, vid):
        # type: (str) -> str
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

    def get_paths(self, elm):
        # type: (str) -> (iter, iter)
        return self.__pm.get_paths(elm)

    def add_seed(self, uri, type):
        # type: (str, str) -> str
        return self.__sm.add_seed(uri, type)

    @property
    def prefixes(self):
        return self.__index.schema.prefixes

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
