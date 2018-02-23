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
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#
"""
import logging

import redis
import sys
from redis.exceptions import BusyLoadingError, RedisError
from time import sleep
from agora.engine.utils import prepare_store_path
import imp

__author__ = 'Fernando Serena'

log = logging.getLogger('agora.engine.utils.kv')


def __check_kv(kv):
    # type: () -> None
    reqs = 0
    while True:
        log.debug('Checking Redis... ({})'.format(reqs))
        reqs += 1
        try:
            kv.echo('echo')
            break
        except BusyLoadingError as e:
            log.warning(e.message)
        except RedisError, e:
            log.error('Redis is not available')
            raise e
    return kv


kvs = []
path_kvs = {}


def get_redis_lite(*args, **kwargs):
    try:
        imp.find_module('redislite')
        import redislite
        return redislite.StrictRedis(*args, **kwargs)
    except ImportError:
        log.error('Redislite module is not available')
        sys.exit(-1)


def get_kv(persist_mode=False, redis_host='localhost', redis_port=6379, redis_db=1, redis_file=None, base='store',
           path='', **kwargs):
    if persist_mode:
        if redis_file is not None:
            prepare_store_path(base, path)
            redis_path = str('/'.join(filter(lambda x: x, [base, path, redis_file])))
            kv = get_redis_lite(redis_path)
            kvs.append(kv)
            path_kvs[redis_path + '.settings'] = kv
            return kv
        else:
            pool = redis.ConnectionPool(host=redis_host, port=redis_port, db=redis_db)
            kv = redis.StrictRedis(connection_pool=pool)
            return __check_kv(kv)
    else:
        return get_redis_lite()


def close():
    for r in kvs:
        kvs.remove(r)
        try:
            r.shutdown()
        except Exception as e:
            print e.message


def close_kv(kv, clear=False):
    if isinstance(kv, str):
        if kv in path_kvs:
            path_kv = kv
            kv = path_kvs[path_kv]
            del path_kvs[path_kv]
        else:
            return False

    if kv in kvs:
        kvs.remove(kv)

    retries = 0
    while retries < 3:
        try:
            kv.shutdown()
            return True
        except Exception as e:
            print e.message
            retries += 1
            sleep(0.5)

    return False
