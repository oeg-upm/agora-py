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
import redislite
from redis.exceptions import BusyLoadingError, RedisError

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


def get_kv(persist_mode=False, redis_host='localhost', redis_port=6379, redis_db=1, redis_file=None):
    if persist_mode:
        if redis_file is not None:
            kv = redislite.StrictRedis(redis_file)
        else:
            pool = redis.ConnectionPool(host=redis_host, port=redis_port, db=redis_db)
            kv = redis.StrictRedis(connection_pool=pool)
    else:
        kv = redislite.StrictRedis()
    return __check_kv(kv)
