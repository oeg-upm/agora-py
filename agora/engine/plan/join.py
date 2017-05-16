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

from rdflib import BNode, Graph
from rdflib import Literal
from rdflib import RDF

from agora import Fountain

__author__ = 'Fernando Serena'

log = logging.getLogger('agora.engine.plan.join')


def _stringify_tp(context, (s, p, o)):
    def stringify_elm(elm):
        if isinstance(elm, BNode):
            return elm.n3(context.namespace_manager)
        elif isinstance(elm, Literal):
            return elm.toPython()

        return context.qname(elm)

    return '{} {} {} .'.format(stringify_elm(s), stringify_elm(p), stringify_elm(o))


def type_hierarchy(fountain, ty):
    type_dict = fountain.get_type(ty)
    return set([ty] + type_dict['sub'])


def subject_join(fountain, tp_paths, context, tp1, tp2, **kwargs):
    # type: (Fountain, iter, Graph, tuple, tuple, dict) -> iter
    subject, pr1, o1 = tp1
    _, pr2, o2 = tp2

    if pr2 == RDF.type:
        o2 = context.qname(o2)
        tp2_domain = [o2]
        tp2_domain.extend(fountain.get_type(o2).get('super'))
    else:
        tp2_domain = fountain.get_property(context.qname(pr2)).get('domain')

    filter_paths = tp_paths[tp1][:]

    if pr1 == RDF.type:
        for path in tp_paths[tp1]:
            steps = path.get('steps')
            if len(steps):
                last_prop = path.get('steps')[-1].get('property')
                dom_r = fountain.get_property(last_prop).get('range')
                if len(filter(lambda x: x in tp2_domain, dom_r)):
                    filter_paths.remove(path)
            else:
                filter_paths.remove(path)
    elif pr2 == RDF.type:
        for path in tp_paths[tp1]:
            last_type = path.get('steps')[-1].get('type')
            types = type_hierarchy(fountain, last_type)
            if set.intersection(types, tp2_domain):
                filter_paths.remove(path)
    else:
        for path in tp_paths[tp1]:
            steps = path.get('steps', [])
            if len(steps):
                matching_steps = steps[:-1]
                for o_path in tp_paths[tp2]:
                    o_steps = o_path.get('steps', [])
                    if len(o_steps):
                        o_matching_steps = o_steps[:-1]
                        if matching_steps == o_matching_steps and path in filter_paths:
                            filter_paths.remove(path)

    return filter_paths


def subject_object_join(fountain, tp_paths, context, tp1, tp2, hints=None, cycles=None):
    # type: (Fountain, iter, Graph, tuple, tuple, dict) -> iter
    subject, pr1, o1 = tp1
    _, pr2, o2 = tp2

    pr2 = context.qname(pr2)
    filter_paths = tp_paths[tp1][:]

    if pr1 == RDF.type or subject == o2:
        for path in tp_paths[tp1]:
            steps = path.get('steps', [])
            if len(steps):
                if pr1 == RDF.type:
                    matching_steps = steps[:]
                else:
                    matching_steps = steps[:-1]
                for o_path in tp_paths[tp2]:
                    if o_path.get('steps') == matching_steps and path in filter_paths:
                        filter_paths.remove(path)
    elif pr2 == context.qname(RDF.type):
        tp1_range = fountain.get_property(context.qname(pr1)).get('range')
        o2 = context.qname(o2)
        for r_type in tp1_range:
            check_types = fountain.get_type(r_type).get('super')
            check_types.append(r_type)
            if o2 in check_types:
                filter_paths = []
                break
        if not filter_paths and hints is not None:
            hints[tp2]['check'] = hints[tp2].get('check', False) or len(tp1_range) > 1
    else:
        if not subject == o2:
            for path in tp_paths[tp1]:
                for o_path in tp_paths[tp2]:
                    steps = o_path.get('steps', [])
                    if len(steps):
                        if pr2 == RDF.type:
                            matching_steps = steps[:]
                        else:
                            matching_steps = steps[:-1]
                        if path.get('steps') == matching_steps and path in filter_paths:
                            filter_paths.remove(path)
    return filter_paths


def object_join(fountain, tp_paths, context, tp1, tp2, **kwargs):
    # type: (Fountain, iter, Graph, tuple, tuple, dict) -> iter
    _, pr1, obj = tp1
    _, pr2, _ = tp2

    tp2_range = fountain.get_property(context.qname(pr2)).get('range')
    tp1_range = fountain.get_property(context.qname(pr1)).get('range')

    if len(filter(lambda x: x in tp1_range, tp2_range)):
        return []

    return tp_paths[tp1]
