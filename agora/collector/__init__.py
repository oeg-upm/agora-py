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

from agora.collector.cache import RedisCache

from agora.engine.plan.graph import AGORA
from rdflib import BNode
from rdflib import Literal
from rdflib import URIRef

from agora.collector.execution import PlanExecutor
from agora.engine.plan.agp import AGP
from shortuuid import uuid

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
    def __init__(self):
        # type: () -> Collector
        self.__cache = None
        self.__planner = None
        self.__loader = None
        self.__force_seed = None

    @property
    def loader(self):
        return self.__loader

    @loader.setter
    def loader(self, l):
        self.__loader = l

    @property
    def force_seed(self):
        return self.__force_seed

    @force_seed.setter
    def force_seed(self, s):
        self.__force_seed = s

    @property
    def planner(self):
        return self.__planner

    @planner.setter
    def planner(self, p):
        self.__planner = p

    @property
    def cache(self):
        return self.__cache

    @cache.setter
    def cache(self, c):
        self.__cache = c

    def get_fragment_generator(self, agp, **kwargs):
        # type: (AGP) -> dict

        force_seed_tuples = []
        if self.__force_seed:
            for ty in self.__force_seed.keys():
                force_seed_tuples.append((URIRef('http://{}'.format(uuid())), ty))
        plan = self.__planner.make_plan(agp, force_seed=force_seed_tuples)

        for s, ty in force_seed_tuples:
            trees = list(plan.subjects(predicate=AGORA.hasSeed, object=s))
            for t in trees:
                plan.remove((t, AGORA.hasSeed, s))
                for actual_s in self.__force_seed[ty]:
                    plan.add((t, AGORA.hasSeed, actual_s))

        executor = PlanExecutor(plan)

        def with_context_ttl():
            return executor.ttl

        def with_context_derefs():
            return executor.n_derefs

        fragment_dict = executor.get_fragment_generator(cache=self.cache, loader=self.__loader, **kwargs)
        fragment_dict['ttl'] = with_context_ttl
        fragment_dict['n_derefs'] = with_context_derefs
        return fragment_dict

    @property
    def prefixes(self):
        return self.planner.fountain.prefixes


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
