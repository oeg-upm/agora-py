# coding=utf-8
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

import calendar
import logging
import math
import shutil
import traceback
from StringIO import StringIO
from threading import Thread, Lock as TLock
from time import sleep

import shortuuid
from agora.collector.execution import parse_rdf
from agora.collector.http import http_get, extract_ttl
from agora.engine.utils import stopped
from agora.engine.utils.graph import get_triple_store
from agora.engine.utils.kv import get_kv
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime as dt, timedelta as delta
from rdflib import ConjunctiveGraph
from rdflib.graph import Graph
from redis.lock import Lock

__author__ = 'Fernando Serena'

log = logging.getLogger('agora.collector.cache')

tpool = ThreadPoolExecutor(max_workers=1)

_caches = {}
_lock = TLock()


def get(**kwargs):
    tuple_id = tuple([kwargs[k] for k in sorted(kwargs.keys())])
    with _lock:
        if tuple_id not in _caches:
            c = RedisCache(**kwargs)
            _caches[tuple_id] = c
        return _caches[tuple_id]


class RedisCache(object):
    def __init__(self, persist_mode=False, key_prefix='', min_cache_time=5, force_cache_time=False,
                 base='store', path='cache', redis_host='localhost', redis_port=6379, redis_db=1, redis_file=None):
        self.__key_prefix = key_prefix
        self.__cache_key = '{}:cache'.format(key_prefix)
        self.__persist_mode = persist_mode
        self.__min_cache_time = min_cache_time
        self.__force_cache_time = force_cache_time
        self.__base_path = base
        self.__resource_cache = get_triple_store()
        self._r = get_kv(persist_mode, redis_host, redis_port, redis_db, redis_file, base=base, path=path)
        self.__lock = Lock(self._r, key_prefix)

        self.__resources_ts = {}

        for lock_key in self._r.keys('{}:l*'.format(self.__key_prefix)):
            self._r.delete(lock_key)

        self.__enabled = True
        self.__purge_th = Thread(target=self.__purge)
        self.__purge_th.daemon = True
        self.__purge_th.start()

    @property
    def r(self):
        return self._r

    @r.setter
    def r(self, r):
        self._r = r

    @property
    def min_cache_time(self):
        return self.__min_cache_time

    @min_cache_time.setter
    def min_cache_time(self, t):
        self.__min_cache_time = t

    @property
    def resource_cache(self):
        return self.__resource_cache

    @resource_cache.setter
    def resource_cache(self, rc):
        self.__resource_cache = rc

    @property
    def persist_mode(self):
        return self.__persist_mode

    @property
    def base_path(self):
        return self.__base_path

    def __clean(self, name):
        shutil.rmtree('{}/{}'.format(self.__base_path, name))

    def uri_lock(self, uri):
        with self.__lock:
            key = '{}:l:'.format(self.__key_prefix) + uri
            key = (key[:250]) if len(key) > 250 else key
            return Lock(self._r, key)

    def __purge(self):
        gids_key = '{}:gids'.format(self.__cache_key)
        while self.__enabled and not stopped.is_set():
            try:
                gids = self._r.hkeys(gids_key)
                obsolete = filter(
                    lambda x: not self._r.exists('{}:{}'.format(self.__cache_key, self._r.hget(gids_key, x))),
                    gids)

                if obsolete:
                    with self._r.pipeline(transaction=True) as p:
                        p.multi()
                        log.debug('Removing {} resouces from cache...'.format(len(obsolete)))
                        for uri in obsolete:
                            with self.uri_lock(uri):
                                uuid = self._r.hget(gids_key, uri)
                                try:
                                    self._r.hdel(gids_key, uri)
                                except Exception:
                                    traceback.print_exc()
                                    log.error('Purging resource {}'.format(uri))
                                p.execute()
            except Exception, e:
                traceback.print_exc()
                log.error(e.message)
                self.__enabled = False
            sleep(10)

    def create(self, conjunctive=False, gid=None, loader=None, format=None):
        if conjunctive:
            uuid = shortuuid.uuid()
            g = get_triple_store(self.__persist_mode, base=self.__base_path, path=uuid)
            return g
        else:
            p = self._r.pipeline(transaction=True)
            p.multi()

            g = Graph(identifier=gid)

            with self.uri_lock(gid):
                uuid = self._r.hget('{}:gids'.format(self.__cache_key), gid)
                if not uuid:
                    uuid = shortuuid.uuid()
                    p.hset('{}:gids'.format(self.__cache_key), gid, uuid)

                gid_key = '{}:{}'.format(self.__cache_key, uuid)

                ttl_ts = self._r.hget(gid_key, 'ttl')
                if ttl_ts is not None:
                    ttl_dt = dt.utcfromtimestamp(int(ttl_ts))
                    now = dt.utcnow()
                    if ttl_dt > now:
                        ttl = math.ceil((ttl_dt - dt.utcnow()).total_seconds())
                        source = self._r.hget(gid_key, 'data')
                        g.parse(StringIO(source), format=format)
                        return g, math.ceil(ttl)

                log.debug('Caching {}'.format(gid))
                response = loader(gid, format)
                if response is None and loader != http_get:
                    response = http_get(gid, format)

                if isinstance(response, bool):
                    return response

                ttl = self.__min_cache_time
                source, headers = response
                if not isinstance(source, Graph):
                    data = source
                    parse_rdf(g, source, format, headers)
                else:
                    if g != source:
                        data = source.serialize(format='turtle')
                        g.__iadd__(source)

                if not self.__force_cache_time:
                    ttl = extract_ttl(headers) or ttl

                p.hset(gid_key, 'data', data)
                ttl_ts = calendar.timegm((dt.utcnow() + delta(seconds=ttl)).timetuple())
                p.hset(gid_key, 'ttl', ttl_ts)
                p.expire(gid_key, ttl)
                p.execute()
            return g, int(ttl)

    def release(self, g):
        if isinstance(g, ConjunctiveGraph):
            if self.__persist_mode:
                # g.close()
                tpool.submit(self.__clean, g.identifier.toPython())
            else:
                g.remove((None, None, None))
                # g.close()

    def close(self):
        self.__enabled = False
