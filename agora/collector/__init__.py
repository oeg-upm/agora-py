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
import re
from abc import abstractmethod

from agora.collector.execution import PlanExecutor
from agora.engine.plan.agp import AGP
from rdflib import BNode
from rdflib import Literal
from rdflib import URIRef

__author__ = "Fernando Serena"

log = logging.getLogger('agora.collector')


class AbstractCollector(object):
    @property
    @abstractmethod
    def prefixes(self):
        raise NotImplementedError

    @abstractmethod
    def get_fragment_generator(self, agp, **kwargs):
        # type: (AGP, dict) -> iter
        raise NotImplementedError


class Collector(AbstractCollector):
    def __init__(self, planner, cache=None):
        # type: (AbstractPlanner, Cache) -> Collector
        self.cache = cache
        self.__planner = planner
        self.__loader = None

    @property
    def loader(self):
        return self.__loader

    @loader.setter
    def loader(self, l):
        self.__loader = l

    @property
    def planner(self):
        return self.__planner

    def get_fragment_generator(self, agp, **kwargs):
        # type: (AGP) -> dict

        plan = self.__planner.make_plan(agp)
        executor = PlanExecutor(plan)

        def with_context():
            return executor.ttl

        fragment_dict = executor.get_fragment_generator(cache=self.cache, loader=self.__loader, **kwargs)
        fragment_dict['ttl'] = with_context
        return fragment_dict

    @property
    def prefixes(self):
        return self.__planner.fountain.prefixes


def triplify(x):
    def __extract_lang(v):
        def __lang_tag_match(strg, search=re.compile(r'[^a-z]').search):
            return not bool(search(strg))

        if '@' in v:
            try:
                (v_aux, lang) = tuple(v.split('@'))
                (v, lang) = (v_aux, lang) if __lang_tag_match(lang) else (v, None)
            except ValueError:
                lang = None
        else:
            lang = None
        return v, lang

    def __term(elm):
        if elm.startswith('<'):
            return URIRef(elm.lstrip('<').rstrip('>'))
        elif '^^' in elm:
            (value, ty) = tuple(elm.split('^^'))
            return Literal(value.replace('"', ''), datatype=URIRef(ty.lstrip('<').rstrip('>')))
        elif elm.startswith('_:'):
            return BNode(elm.replace('_:', ''))
        else:
            (elm, lang) = __extract_lang(elm)
            elm = elm.replace('"', '')
            if lang is not None:
                return Literal(elm, lang=lang)
            else:
                return Literal(elm)

    c, s, p, o = eval(x)
    return c, __term(s), __term(p), __term(o)
