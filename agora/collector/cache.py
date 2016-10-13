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

import logging
import shutil
import traceback
from threading import Thread

import shortuuid
from agora.engine.utils.graph import get_triple_store
from agora.engine.utils.kv import get_kv
from concurrent.futures import ThreadPoolExecutor
from rdflib import ConjunctiveGraph
from rdflib.graph import Graph
from redis.lock import Lock as RedisLock
from time import sleep
from werkzeug.http import parse_dict_header

__author__ = 'Fernando Serena'

log = logging.getLogger('agora.collector.cache')

tpool = ThreadPoolExecutor(max_workers=4)


class RedisCache(object):
    def __init__(self, persist_mode=False, key_prefix='', min_cache_time=5,
                 base='store', path='cache', redis_host='localhost', redis_port=6379, redis_db=1, redis_file=None):
        self.__key_prefix = key_prefix
        self.__cache_key = '{}:cache'.format(key_prefix)
        self.__gids_key = '{}:gids'.format(self.__cache_key)
        self.__persist_mode = persist_mode
        self.__min_cache_time = min_cache_time
        self.__base_path = base
        self.__resource_path = path
        self.__resource_cache = get_triple_store(persist_mode=persist_mode,
                                                 base=base, path=path)
        self._r = get_kv(persist_mode, redis_host, redis_port, redis_db, redis_file)

        self.__clear_cache()
        self.__resources_ts = {}

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
        self.__clear_cache()

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

    def __clear_cache(self):
        cache_keys = list(self._r.keys('{}*lock*'.format(self.__key_prefix)))
        with self._r.pipeline(transaction=True) as p:
            for key in cache_keys:
                p.delete(key)
            p.execute()
        log.debug('Cleared {} keys'.format(len(cache_keys)))
        # rmtree(self.__base_path)

    def __clean(self, name):
        shutil.rmtree('{}/{}'.format(self.__base_path, name))

    def gid_lock(self, uuid):
        lock_key = '{}:cache:{}:lock'.format(self.__key_prefix, uuid)
        return self._r.lock(lock_key, lock_class=RedisLock)

    def __purge(self):
        while self.__enabled:
            try:
                obsolete = filter(lambda x: not self._r.exists('{}:cache:{}'.format(self.__key_prefix, x)),
                                  self._r.smembers(self.__cache_key))

                if obsolete:
                    with self._r.pipeline(transaction=True) as p:
                        p.multi()
                        log.debug('Removing {} resouces from cache...'.format(len(obsolete)))
                        for gid in obsolete:
                            gid_lock = self.gid_lock(gid)
                            with gid_lock:
                                counter_key = '{}:cache:{}:cnt'.format(self.__key_prefix, gid)
                                usage_counter = self._r.get(counter_key)
                                if usage_counter is None or int(usage_counter) <= 0:
                                    try:
                                        g = self.__resource_cache.get_context(gid)
                                        self.__resource_cache.remove_context(g)
                                        p.srem(self.__cache_key, gid)
                                        p.delete(counter_key)
                                    except Exception, e:
                                        traceback.print_exc()
                                        log.error('Purging resource {}'.format(gid))
                                p.execute()
            except Exception, e:
                traceback.print_exc()
                log.error(e.message)
            sleep(1)

    def create(self, conjunctive=False, gid=None, loader=None, format=None):
        p = self._r.pipeline(transaction=True)
        p.multi()

        if conjunctive:
            uuid = shortuuid.uuid()
            g = get_triple_store(self.__persist_mode, base=self.__base_path, path=uuid)
            return g
        else:
            temp_key = '{}:{}'.format(self.__cache_key, gid)
            counter_key = '{}:cnt'.format(temp_key)

            lock = self.gid_lock(gid)
            with lock:
                g = self.__resource_cache.get_context(gid)
                cached = self._r.sismember(self.__cache_key, gid)
                if cached:
                    ttl = self._r.get(temp_key)
                    if ttl is not None:
                        p.incr(counter_key)
                        p.execute()
                        return g, int(ttl)
                    else:
                        p.srem(self.__cache_key, gid)
                        p.delete(counter_key)
                        p.execute()

                log.debug(u'Caching {}'.format(gid))
                response = loader(gid, format)
                if isinstance(response, bool):
                    return response

                ttl = self.__min_cache_time
                source, headers = response
                if not isinstance(source, Graph):
                    try:
                        g.parse(source=source, format=format)
                    except Exception as e:
                        print e.message

                else:
                    if g != source:
                        g.__iadd__(source)

                cache_control = headers.get('Cache-Control', None)
                if cache_control is not None:
                    cache_dict = parse_dict_header(cache_control)
                    ttl = int(cache_dict.get('max-age', ttl))

                # Let's create a new one
                p.delete(counter_key)
                p.sadd(self.__cache_key, gid)
                p.incr(counter_key)

                p.set(temp_key, ttl)
                p.expire(temp_key, ttl)
                p.execute()

                return g, ttl

    def release(self, g):
        if isinstance(g, ConjunctiveGraph):
            if self.__persist_mode:
                g.close()
                tpool.submit(self.__clean, g.identifier.toPython())
            else:
                g.remove((None, None, None))
                g.close()
        else:
            gid = g.identifier.toPython()
            with self.gid_lock(gid):
                if self._r.sismember(self.__cache_key, gid):
                    self._r.decr('{}:{}:cnt'.format(self.__cache_key, gid))

    def close(self):
        self.__enabled = False
        tpool.shutdown(wait=True)

    def __del__(self):
        self.close()
