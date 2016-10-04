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
import shutil
import traceback
from threading import Lock, Thread

import redislite
import shortuuid
from agora.engine.utils.graph import get_resource_cache, rmtree
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime as dt, timedelta as delta
from rdflib import ConjunctiveGraph
from redis.lock import Lock as RedisLock
from time import sleep
from werkzeug.http import parse_dict_header

__author__ = 'Fernando Serena'


log = logging.getLogger('agora.collector.cache')


class RedisCache(object):
    def __init__(self, persist_mode=False, key_prefix='', min_cache_time=5,
                 base='cache', path='resources'):
        self.__last_creation_ts = dt.utcnow()
        self.__graph_dict = {}
        self.__uuid_dict = {}
        self.__gid_uuid_dict = {}
        self.__lock = Lock()
        self.__key_prefix = key_prefix
        self.__cache_key = '{}:cache'.format(key_prefix)
        self.__gids_key = '{}:cache:gids'.format(self.__cache_key)
        self.__persist_mode = persist_mode
        self.__min_cache_time = min_cache_time
        self.__base_path = base
        self._r = redislite.StrictRedis()

        self.__clear_cache()
        self.__resources_ts = {}
        self.__resource_cache = get_resource_cache(persist_mode=persist_mode,
                                                   base=base, path=path)

        self.__pool = ThreadPoolExecutor(max_workers=4)
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

    def __clear_cache(self):
        cache_keys = list(self._r.keys('{}:cache*'.format(self.__key_prefix)))
        with self._r.pipeline(transaction=True) as p:
            for key in cache_keys:
                p.delete(key)
            p.execute()
        log.debug('Cleared {} keys'.format(len(cache_keys)))
        rmtree(self.__base_path)

    @staticmethod
    def __clean(name):
        shutil.rmtree('store/resources/{}'.format(name))

    def uuid_lock(self, uuid):
        lock_key = '{}:cache:{}:lock'.format(self.__key_prefix, uuid)
        return self._r.lock(lock_key, lock_class=RedisLock)

    def __purge(self):
        while True:
            self.__lock.acquire()
            try:
                obsolete = filter(lambda x: not self._r.exists('{}:cache:{}'.format(self.__key_prefix, x)),
                                  self._r.smembers(self.__cache_key))

                if obsolete:
                    with self._r.pipeline(transaction=True) as p:
                        p.multi()
                        log.debug('Removing {} resouces from cache...'.format(len(obsolete)))
                        for uuid in obsolete:
                            uuid_lock = self.uuid_lock(uuid)
                            uuid_lock.acquire()
                            try:
                                gid = self._r.hget(self.__gids_key, uuid)
                                counter_key = '{}:cache:{}:cnt'.format(self.__key_prefix, uuid)
                                usage_counter = self._r.get(counter_key)
                                if usage_counter is None or int(usage_counter) <= 0:
                                    try:
                                        self.__resource_cache.remove_context(self.__resource_cache.get_context(uuid))
                                        p.srem(self.__cache_key, uuid)
                                        p.hdel(self.__gids_key, uuid)
                                        p.hdel(self.__gids_key, gid)
                                        p.delete(counter_key)
                                        g = self.__uuid_dict.get(uuid, None)
                                        if g is not None:
                                            del self.__uuid_dict[uuid]
                                            del self.__graph_dict[g]
                                    except Exception, e:
                                        traceback.print_exc()
                                        log.error('Purging resource {} with uuid {}'.format(gid, uuid))
                                p.execute()
                            finally:
                                uuid_lock.release()
            except Exception, e:
                traceback.print_exc()
                log.error(e.message)
            finally:
                self.__lock.release()
            sleep(1)

    def create(self, conjunctive=False, gid=None, loader=None, format=None):
        lock = None
        cached = False
        temp_key = None
        p = self._r.pipeline(transaction=True)
        p.multi()

        uuid = shortuuid.uuid()

        if conjunctive:
            g = get_resource_cache(self.__persist_mode, base=self.__base_path, path=uuid)
            # if self.__persist_mode:
            #     g = ConjunctiveGraph('Sleepycat')
            #     g.open('store/resources/{}'.format(uuid), create=True)
            # else:
            #     g = ConjunctiveGraph()
            # g.store.graph_aware = False
            self.__graph_dict[g] = uuid
            self.__uuid_dict[uuid] = g
            return g
        else:
            g = None
            try:
                st_uuid = self._r.hget(self.__gids_key, gid)
                if st_uuid is not None:
                    cached = True
                    uuid = st_uuid
                    lock = self.uuid_lock(uuid)
                    lock.acquire()
                    g = self.__uuid_dict.get(uuid, None)
                    lock.release()

                if st_uuid is None or g is None:
                    st_uuid = None
                    cached = False
                    uuid = shortuuid.uuid()
                    g = self.__resource_cache.get_context(uuid)

                temp_key = '{}:cache:{}'.format(self.__key_prefix, uuid)
                counter_key = '{}:cnt'.format(temp_key)

                if st_uuid is None:
                    p.delete(counter_key)
                    p.hset(self.__gids_key, uuid, gid)
                    p.hset(self.__gids_key, gid, uuid)
                    p.sadd(self.__cache_key, uuid)
                    p.set(temp_key, '')  # Prepare temporal key to avoid race conditions on purging
                    p.execute()
                    self.__last_creation_ts = dt.utcnow()
                    p.incr(counter_key)
                lock = self.uuid_lock(uuid)
                lock.acquire()
            except Exception, e:
                log.error(e.message)
                traceback.print_exc()
        if g is not None:
            self.__graph_dict[g] = uuid
            self.__uuid_dict[uuid] = g

        try:
            if cached:
                return g

            log.debug('Caching {}'.format(gid.encode('utf8', 'replace')))
            response = loader(gid, format)
            if not isinstance(response, bool):
                source, headers = response
                g.parse(source=source, format=format)
                if not self._r.get(temp_key):
                    cache_control = headers.get('Cache-Control', None)
                    ttl = self.__min_cache_time
                    if cache_control is not None:
                        cache_dict = parse_dict_header(cache_control)
                        ttl = int(cache_dict.get('max-age', ttl))
                    ttl_ts = calendar.timegm((dt.utcnow() + delta(ttl)).timetuple())
                    p.set(temp_key, ttl_ts)
                    p.expire(temp_key, ttl)
                    p.execute()

                return g
            else:
                p.hdel(self.__gids_key, gid)
                p.hdel(self.__gids_key, uuid)
                p.srem(self.__cache_key, uuid)
                counter_key = '{}:cache:{}:cnt'.format(self.__key_prefix, uuid)
                p.delete(counter_key)
                p.execute()
                del self.__graph_dict[g]
                del self.__uuid_dict[uuid]
                return response
        finally:
            p.execute()
            if lock is not None:
                lock.release()

    def release(self, g):
        lock = None
        try:
            if g in self.__graph_dict:
                if isinstance(g, ConjunctiveGraph):
                    if self.__persist_mode:
                        g.close()
                        self.__pool.submit(self.__clean, self.__graph_dict[g])
                    else:
                        g.remove((None, None, None))
                        g.close()
                else:
                    uuid = self.__graph_dict[g]
                    if uuid is not None:
                        lock = self.uuid_lock(uuid)
                        lock.acquire()
                        if self._r.sismember(self.__cache_key, uuid):
                            self._r.decr('{}:cache:{}:cnt'.format(self.__key_prefix, uuid))
        finally:
            if lock is not None:
                lock.release()

    def __delete_linked_resource(self, g, subject):
        for (s, p, o) in g.triples((subject, None, None)):
            self.__delete_linked_resource(g, o)
            g.remove((s, p, o))
