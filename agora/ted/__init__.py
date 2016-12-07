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
import shortuuid

from agora.collector.wrapper import ResourceWrapper

__author__ = 'Fernando Serena'


class TED(object):
    def __init__(self):
        self.__ecosystem = Ecosystem()

    @property
    def ecosystem(self):
        return self.__ecosystem


class Ecosystem(object):
    def __init__(self):
        self.__things = set([])

    @property
    def things(self):
        return iter(self.__things)


class Thing(object):
    def __init__(self):
        self.__interactions = set([])

    @property
    def interactions(self):
        return iter(self.__interactions)


class Interaction(object):
    def __init__(self):
        self.__endpoints = set([])

    @property
    def endpoints(self):
        return iter(self.__endpoints)


class Endpoint(object):
    def __init__(self):
        pass


def create_wrapper(ted):
    # type: (TED) -> ResourceWrapper
    wrapper = ResourceWrapper()

    i = 0
    for thing in ted.ecosystem.things:
        for interaction in thing.interactions:
            print interaction
            wrapper.intercept('/<{}>'.format(i))

    return wrapper
