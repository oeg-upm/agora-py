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
import os
from threading import Lock, Event

__author__ = 'Fernando Serena'

stopped = Event()


class Singleton(type):
    _lock = Lock()

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            id = kwargs.get('id', '')
            if id not in cls.instances:
                cls.instances[id] = super(Singleton, cls).__call__(*args, **kwargs)
            return cls.instances[id]


def prepare_store_path(base, path):
    if not os.path.exists(base):
        os.makedirs(base)
    if not os.path.exists('{}/{}'.format(base, path)):
        os.makedirs('{}/{}'.format(base, path))


class Semaphore(object):
    def __init__(self):
        self.value = -1

    def __enter__(self):
        self.value = 0

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.value = 1
        
    def isSet(self):
        return self.value == 1

    def set(self):
        self.value = 1


def get_immediate_subdirectories(a_dir):
    if not os.path.exists(a_dir):
        os.makedirs(a_dir)
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]


class Wrapper(object):
    def __init__(self, object):
        self.__object = object
        self.__argret = {}

    def decorate(self, f):
        def wrapper(*args, **kwargs):
            ret = None
            if kwargs or args not in self.__argret:
                ret = f(*args, **kwargs)

            if kwargs:
                return ret

            if ret is not None:
                self.__argret[args] = ret
            return self.__argret.get(args, None)

        return wrapper

    def __getattr__(self, item):
        if item not in self.__argret:
            attr = self.__object.__getattribute__(item)
            if hasattr(attr, '__call__'):
                return self.decorate(attr)
            else:
                self.__argret[item] = attr

        return self.__argret[item]
