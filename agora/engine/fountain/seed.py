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

import base64
import collections
import logging

import redis
from rfc3987 import parse

from agora.engine.fountain.index import Index
from agora.engine.utils.cache import Cache

__author__ = 'Fernando Serena'

log = logging.getLogger('agora.engine.fountain.seed')


class SeedManager(object):
    def __init__(self):
        # type: () -> SeedManager
        self.__index = None
        self.__cache = Cache()

    @property
    def index(self):
        return self.__index

    @index.setter
    def index(self, i):
        self.__index = i
        self.__cache.watch(self.__index.schema.cache)

    def add_seed(self, uri, type):
        # type: (str, str) -> str
        sid = _add_seed(self.__index.r, uri, type)
        self.__cache.clear()
        return sid

    def delete_seed(self, sid):
        # type: (str) -> None
        _delete_seed(self.__index.r, sid)
        self.__cache.clear()

    def get_seed(self, sid):
        # type: (str) -> dict
        return _get_seed(self.__index.r, sid)

    def delete_type_seeds(self, type):
        # type: (str) -> None
        _delete_type_seeds(self.__index.r, type)
        self.__cache.clear()

    @property
    def seeds(self):
        # type: () -> dict
        return _get_seeds(self.__index.r)

    def get_type_seeds(self, type):
        # type: (str) -> iter
        return _get_type_seeds(self.__index, type)

    def validate(self):
        seed_dict = self.seeds
        types = self.__index.types
        obsolete_types = set.difference(set(seed_dict.keys()), set(types))
        for t in obsolete_types:
            self.delete_type_seeds(t)


class DuplicateSeedError(Exception):
    pass


class InvalidSeedError(Exception):
    pass


def _add_seed(r, uri, ty):
    # type: (redis.StrictRedis, str, str) -> str
    parse(uri, rule='URI')
    type_found = False
    type_keys = r.keys('*:types')
    for tk in type_keys:
        if r.sismember(tk, ty):
            type_found = True
            encoded_uri = base64.b64encode(uri)
            if r.sismember('seeds:{}'.format(ty), encoded_uri):
                raise DuplicateSeedError('{} is already registered as a seed of type {}'.format(uri, ty))
            r.sadd('seeds:{}'.format(ty), base64.b64encode(uri))
            break

    if not type_found:
        raise TypeError("{} is not a valid type".format(ty))

    return base64.b64encode('{}|{}'.format(ty, uri))


def _get_seed(r, sid):
    # type: (redis.StrictRedis, str) -> dict
    try:
        ty, uri = base64.b64decode(sid).split('|')
        if r.sismember('seeds:{}'.format(ty), base64.b64encode(uri)):
            return {'type': ty, 'uri': uri}
    except (ValueError, TypeError) as e:
        raise InvalidSeedError(e.message)

    raise InvalidSeedError(sid)


def _delete_seed(r, sid):
    # type: (redis.StrictRedis, str) -> None
    try:
        ty, uri = base64.b64decode(sid).split('|')
        set_key = 'seeds:{}'.format(ty)
        encoded_uri = base64.b64encode(uri)
        if not r.srem(set_key, encoded_uri):
            raise InvalidSeedError(sid)
    except TypeError as e:
        raise InvalidSeedError(e.message)


def _delete_type_seeds(r, ty):
    # type: (redis.StrictRedis, str) -> None
    r.delete('seeds:{}'.format(ty))


def _get_seeds(r):
    # type: (redis.StrictRedis) -> dict
    def iterator():
        seed_types = r.keys('seeds:*')
        for st in seed_types:
            ty = st.replace('seeds:', '')
            for seed in list(r.smembers(st)):
                yield ty, base64.b64decode(seed)

    result_dict = collections.defaultdict(list)
    for t, uri in iterator():
        result_dict[t].append({"uri": uri, "id": base64.b64encode('{}|{}'.format(t, uri))})
    return result_dict


def _get_type_seeds(index, ty):
    # type: (Index) -> iter
    try:
        t_dict = index.get_type(ty)
        all_seeds = set([])
        for t in t_dict['sub'] + [ty]:
            all_seeds.update([base64.b64decode(seed) for seed in list(index.r.smembers('seeds:{}'.format(t)))])
        return list(all_seeds)
    except TypeError:
        # Check if it is a property...and return an empty list
        try:
            index.get_property(ty)
            return []
        except TypeError:
            raise TypeError(ty)
