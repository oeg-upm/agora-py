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
import traceback
from abc import abstractmethod
from datetime import datetime

import agora.engine.plan.join
from agora.engine.plan.agp import AgoraGP
from agora.engine.plan.graph import graph_plan
from rdflib import RDF

__author__ = 'Fernando Serena'

log = logging.getLogger('agora.engine.plan')


def _get_tp_paths(fountain, agp):
    def __join(f, joins):
        invalid_paths = []
        for (sj, pj, oj) in joins:
            invalid_paths.extend(f(fountain, tp_paths, c, (s, pr, o), (sj, pj, oj), hints=tp_hints))
        if len(joins):
            tp_paths[(s, pr, o)] = filter(lambda z: z not in invalid_paths, tp_paths[(s, pr, o)])
        join_paths.extend(invalid_paths)

    tp_paths = {}
    tp_hints = {}
    for c in agp.contexts():
        for (s, pr, o) in c.triples((None, None, None)):
            tp_hints[(s, pr, o)] = {}
            try:
                if pr == RDF.type:
                    tp_paths[(s, pr, o)] = fountain.get_paths(agp.qname(o))['paths']
                else:
                    tp_paths[(s, pr, o)] = fountain.get_paths(agp.qname(pr))['paths']
            except IOError as e:
                raise NameError('Cannot get a path to an unknown subject: {}'.format(e.message))

        while True:
            join_paths = []

            for (s, pr, o) in c.triples((None, None, None)):
                if len(tp_paths[(s, pr, o)]):
                    s_join = [(x, pj, y) for (x, pj, y) in c.triples((s, None, None)) if pj != pr]
                    __join(join.subject_join, s_join)
                    o_join = [(x, pj, y) for (x, pj, y) in c.triples((None, None, o)) if pj != pr]
                    __join(join.object_join, o_join)
                    so_join = [(x, pj, y) for (x, pj, y) in c.triples((None, None, s))]
                    so_join.extend([(x, pj, y) for (x, pj, y) in c.triples((o, None, None))])
                    __join(join.subject_object_join, so_join)
            if not join_paths:
                break

    for (s, pr, o) in tp_hints:
        if pr == RDF.type and 'check' not in tp_hints[(s, pr, o)]:
            tp_hints[(s, pr, o)]['check'] = len(fountain.get_type(agp.qname(o)).get('super')) > 0

    return tp_paths, tp_hints


class Plan(object):
    def __init__(self, fountain, *tps):
        # type: (Fountain, list) -> Plan
        self.__fountain = fountain
        gp = '{ %s }' % ' . '.join(tps)
        agora_gp = AgoraGP.from_string(gp, fountain.prefixes)
        if agora_gp is None:
            raise AttributeError('{} is not a valid graph pattern'.format(gp))

        log.debug('Agora Graph Pattern:\n{}'.format(agora_gp.graph.serialize(format='turtle')))

        try:
            search, hints = _get_tp_paths(fountain, agora_gp.graph)
            self.__plan = {
                "plan": [{"context": agora_gp.get_tp_context(tp), "pattern": tp, "paths": paths, "hints": hints[tp]}
                         for (tp, paths) in search.items()], "prefixes": agora_gp.prefixes}

            self.__g_plan = graph_plan(self.__plan, self.__fountain)
        except TypeError, e:
            raise NameError(e.message)

    @property
    def json(self):
        return self.__plan

    @property
    def graph(self):
        return self.__g_plan


class AbstractPlanner(object):
    @property
    @abstractmethod
    def fountain(self):
        # type: () -> Fountain
        raise NotImplementedError

    def make_plan(self, *tps):
        # type: (list) -> Graph
        raise NotImplementedError


class Planner(AbstractPlanner):
    def __init__(self, fountain):
        # type: (Fountain) -> Planner
        self.__fountain = fountain

    @property
    def fountain(self):
        return self.__fountain

    def make_plan(self, *tps):
        plan = Plan(self.__fountain, *tps)
        return plan.graph
