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
from datetime import datetime as dt, timedelta as delta
from threading import Thread, Lock
from time import sleep

import shortuuid
from agora.engine.utils.graph import get_triple_store
from agora.engine.utils.kv import get_kv
from concurrent.futures import ThreadPoolExecutor
from rdflib import ConjunctiveGraph
from rdflib.graph import Graph
from werkzeug.http import parse_dict_header

__author__ = 'Fernando Serena'

log = logging.getLogger('agora.collector.cache')

tpool = ThreadPoolExecutor(max_workers=1)


class RedisCache(object):
    def __init__(self, persist_mode=False, key_prefix='', min_cache_time=5,
                 base='store', path='cache', redis_host='localhost', redis_port=6379, redis_db=1, redis_file=None):
        self.__key_prefix = key_prefix
        self.__cache_key = '{}:cache'.format(key_prefix)
        self.__locks = {}
        self.__lock = Lock()
        self.__uuids = {}
        self.__persist_mode = persist_mode
        self.__min_cache_time = min_cache_time
        self.__base_path = base
        self.__resource_path = path
        self.__resource_cache = get_triple_store(persist_mode=persist_mode,
                                                 base=base, path=path)
        self._r = get_kv(persist_mode, redis_host, redis_port, redis_db, redis_file)

        self.__resources_ts = {}

        cached_uris = 0
        for uuid_key in self._r.keys('{}:*uri'.format(key_prefix)):
            uuid = uuid_key.split(':')[1]
            uri = self._r.get(uuid_key)
            if uri is not None:
                self.__uuids[uri] = uuid
                cached_uris += 1

        log.info('Recovered {} cached resources'.format(cached_uris))

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

    def gid_lock(self, gid):
        with self.__lock:
            if gid not in self.__locks:
                self.__locks[gid] = Lock()
            return self.__locks[gid]

    def __purge(self):
        while self.__enabled:
            try:
                obsolete = filter(
                    lambda x: not self._r.exists('{}:cache:{}'.format(self.__key_prefix, x)),
                    self._r.smembers(self.__cache_key))

                if obsolete:
                    with self._r.pipeline(transaction=True) as p:
                        p.multi()
                        log.debug('Removing {} resouces from cache...'.format(len(obsolete)))
                        for uuid in obsolete:
                            uri_key = '{}:{}:uri'.format(self.__key_prefix, uuid)
                            gid = self._r.get(uri_key)
                            with self.gid_lock(gid):
                                try:
                                    g = self.__resource_cache.get_context(gid)
                                    g.remove((None, None, None))
                                    self.__resource_cache.remove_context(g)
                                    p.srem(self.__cache_key, uuid)
                                    p.delete(uri_key)
                                    with self.__lock:
                                        del self.__locks[gid]
                                except Exception, e:
                                    traceback.print_exc()
                                    log.error('Purging resource {}'.format(gid))
                                p.execute()
            except Exception, e:
                traceback.print_exc()
                log.error(e.message)
            sleep(1)

    def create(self, conjunctive=False, gid=None, loader=None, format=None):
        if conjunctive:
            uuid = shortuuid.uuid()
            g = get_triple_store(self.__persist_mode, base=self.__base_path, path=uuid)
            return g
        else:
            p = self._r.pipeline(transaction=True)
            p.multi()

            with self.gid_lock(gid):
                g = self.__resource_cache.get_context(gid)
                if gid not in self.__uuids:
                    self.__uuids[gid] = shortuuid.uuid()

                temp_key = '{}:{}'.format(self.__cache_key, self.__uuids[gid])

                ttl_ts = self._r.get(temp_key)
                if ttl_ts is not None:
                    ttl = 0
                    if ttl_ts is not None:
                        ttl_dt = dt.utcfromtimestamp(int(ttl_ts))
                        now = dt.utcnow()
                        if ttl_dt > now:
                            ttl = math.ceil((ttl_dt - dt.utcnow()).total_seconds())

                    g_copy = Graph(identifier=gid)
                    g_copy.__iadd__(g)
                    return g_copy, math.ceil(ttl)

                g.remove((None, None, None))
                self.__resource_cache.remove_context(g)
                g = self.__resource_cache.get_context(gid)

                log.debug('Caching {}'.format(gid))
                response = loader(gid, format)
                if isinstance(response, bool):
                    return response

                ttl = self.__min_cache_time
                source, headers = response
                if not isinstance(source, Graph):
                    try:
                        g.parse(source=source, format=format)
                    except Exception as e:
                        traceback.print_exc()
                        log.error(e.message)

                else:
                    if g != source:
                        g.__iadd__(source)

                cache_control = headers.get('Cache-Control', None)
                if cache_control is not None:
                    cache_dict = parse_dict_header(cache_control)
                    ttl = int(math.ceil(float(cache_dict.get('max-age', ttl))))

                # Let's create a new one
                p.sadd(self.__cache_key, self.__uuids[gid])
                p.set('{}:{}:uri'.format(self.__key_prefix, self.__uuids[gid]), gid)

                ttl_ts = calendar.timegm((dt.utcnow() + delta(seconds=ttl)).timetuple())
                p.set(temp_key, ttl_ts)
                p.expire(temp_key, ttl)
                p.execute()
                g_copy = Graph(identifier=gid)
                g_copy.__iadd__(g)
                return g_copy, int(ttl)

    def release(self, g):
        if isinstance(g, ConjunctiveGraph):
            if self.__persist_mode:
                g.close()
                tpool.submit(self.__clean, g.identifier.toPython())
            else:
                g.remove((None, None, None))
                g.close()

    def close(self):
        self.__enabled = False
        tpool.shutdown(wait=True)

    def __del__(self):
        self.close()
