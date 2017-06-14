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
from threading import Event, Lock

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
