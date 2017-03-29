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
from abc import abstractmethod

from rdflib import RDF

import agora.engine.plan.join
from agora.engine.plan.agp import AGP
from agora.engine.plan.graph import graph_plan, __extend_uri
from agora.engine.utils.lists import subfinder

__author__ = 'Fernando Serena'

log = logging.getLogger('agora.engine.plan')


def extend_with_cycles(tps, paths, cycles, prefixes):
    extended_paths = paths[:]
    new_paths = []
    tps_props = map(lambda x: x[1], tps)
    for path in paths:
        steps = path.get('steps', [])
        path_cycles = path.get('cycles', [])
        for cycle_id in path_cycles:
            cycle_steps = map(lambda c: c['steps'], filter(lambda x: x['cycle'] == cycle_id, cycles))
            for cs in filter(lambda x: not subfinder(steps, x), cycle_steps):
                if not any(map(lambda s: __extend_uri(prefixes, s['property']) in tps_props, cs)):
                    continue
                cycle_root = cs[0]['type']
                try:
                    index = (i for i, v in enumerate(steps) if v['type'] == cycle_root).next()
                    new_path = path.copy()
                    new_path['steps'] = steps[:index] + cs + steps[index:]
                    if new_path not in extended_paths and new_path not in new_paths:
                        new_paths.append(new_path)
                except StopIteration:
                    pass
    extended_paths += new_paths

    return extended_paths


def _get_tp_paths(fountain, agp):
    def __join(f, joins):
        invalid_paths = []
        for (sj, pj, oj) in joins:
            invalid_paths.extend(f(fountain, tp_paths, c, (s, pr, o), (sj, pj, oj), hints=tp_hints, cycles=tp_cycles))
        if len(joins):
            tp_paths[(s, pr, o)] = filter(lambda z: z not in invalid_paths, tp_paths[(s, pr, o)])
        join_paths.extend(invalid_paths)

    prefixes = fountain.prefixes
    tp_paths = {}
    tp_hints = {}
    tp_cycles = {}
    source_tp_paths = {}
    for c in agp.contexts():
        for (s, pr, o) in c.triples((None, None, None)):
            tp_hints[(s, pr, o)] = {}
            try:
                if pr == RDF.type:
                    elm = o
                else:
                    elm = pr

                tp = (s, pr, o)

                comp_paths = fountain.get_paths(agp.qname(elm))
                tp_paths[tp] = comp_paths['paths']
                source_tp_paths[tp] = comp_paths['paths']
                tp_cycles[tp] = comp_paths['all-cycles']
            except IOError as e:
                raise NameError('Cannot get a path to an unknown subject: {}'.format(e.message))

        for tp in tp_paths:
            tp_paths[tp] = extend_with_cycles(filter(lambda x: x != tp, tp_paths.keys()), tp_paths[tp], tp_cycles[tp], prefixes)

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

    return tp_paths, tp_hints, tp_cycles


class Plan(object):
    def __init__(self, fountain, agp):
        # type: (Fountain, AGP) -> Plan
        self.__fountain = fountain
        log.debug('Agora Graph Pattern:\n{}'.format(agp.graph.serialize(format='turtle')))

        try:
            search, hints, cycles = _get_tp_paths(fountain, agp.graph)
            self.__plan = {
                "plan": [{"context": agp.get_tp_context(tp), "pattern": tp, "paths": paths, "hints": hints[tp],
                          "cycles": cycles[tp]}
                         for (tp, paths) in search.items()], "prefixes": agp.prefixes}

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

    def make_plan(self, agp):
        # type: (AGP) -> Graph
        plan = Plan(self.__fountain, agp)
        return plan.graph
