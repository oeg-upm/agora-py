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

import copy
import logging
import traceback
from abc import abstractmethod

import networkx as nx
from rdflib import RDF, URIRef, Variable, BNode, Graph

from agora.engine.fountain import AbstractFountain
from agora.engine.plan.agp import AGP, TP
from agora.engine.plan.graph import graph_plan, __extend_uri

__author__ = 'Fernando Serena'

log = logging.getLogger('agora.engine.plan')


def _root_types(fountain, root_tps, graph):
    root_types = {}
    roots_dict = {}
    for (s, p, o) in root_tps:
        tp_types = set()
        if p == RDF.type:
            if not isinstance(o, URIRef):
                tp_types.update(set(fountain.types))
            else:
                t = graph.qname(o)
                tp_types.add(t)
                tp_types.update(set(fountain.get_type(t)['sub']))
        else:
            tp_types.update(set(fountain.get_property(graph.qname(p))['domain']))

        roots_dict[(s, p, o)] = tp_types

    for (s, p, o) in roots_dict:
        root_join = [(sr, pr, orr) for (sr, pr, orr) in roots_dict if sr == s]
        types = [roots_dict[(s, p, o)]]
        for (sr, pr, orr) in root_join:
            types.append(roots_dict[(sr, pr, orr)])
        types = set.intersection(*types)
        if types:
            root_types[(s, p, o)] = types
    return root_types


def _validate_agp_context(fountain, c):
    s_join_tested = []
    for s, pr, o in c.triples((None, None, None)):
        if pr == RDF.type:
            continue
        if (s, pr, o) not in s_join_tested:
            s_join = [(x, pj, y) for (x, pj, y) in c.triples((s, None, None)) if pj != pr]
            root_types = _root_types(fountain, s_join + [(s, pr, o)], c)
            s_join_tested.append((s, pr, o))
            if not root_types:
                raise TypeError('bad subject join')
        so_join = list(c.triples((o, None, None)))
        pr_dict = fountain.get_property(c.qname(pr))
        for (so, pro, oo) in so_join:
            if pro == RDF.type:
                if isinstance(so, URIRef):
                    pro_domain = {c.qname(oo)}
                else:
                    pro_domain = set(fountain.types)
            else:
                pro_domain = set(fountain.get_property(c.qname(pro))['domain'])
            if not set.intersection(pro_domain, set(pr_dict['range'])):
                raise TypeError('bad subject-object join')
        o_join = [(x, pj, y) for (x, pj, y) in c.triples((None, None, o)) if pj != pr]
        for (so, pro, oo) in o_join:
            if pro == RDF.type:
                pro_range = set(fountain.types)
            else:
                pro_range = set(fountain.get_property(c.qname(pro))['range'])
            if not set.intersection(pro_range, set(pr_dict['range'])):
                raise TypeError('bad object-object join')


def _chunks(l, n):
    for i in xrange(0, len(l), n):
        yield l[i:i + n]


def _build_root_paths(fountain, path, rt):
    def _continue(root, steps=None, index=0, last_pr=None, last_ty=None, last_pr_dict=None):
        if steps is None:
            steps = []

        if index >= len(path):
            yield steps
        else:
            index_pr = path[index]
            if index == 0:
                step = {'type': root, 'property': index_pr}
                steps.append(step)
                for p in _continue(root, steps, index=1, last_pr=index_pr, last_ty=root):
                    yield p
            else:
                if last_pr_dict is None:
                    last_pr_dict = fountain.get_property(last_pr)

                if index_pr != 'rdf:type':
                    pr_dict = fountain.get_property(index_pr)
                    last_pr_constraints = last_pr_dict['constraints']
                    expected = set.intersection(set(pr_dict['domain']), set(last_pr_dict['range']))
                    expected = filter(lambda x: not set.intersection(set(fountain.get_type(x)['super']), expected),
                                      expected)
                else:
                    expected = set(last_pr_dict['range'])  # no more iterations from here!
                    last_pr_constraints = []
                for et in expected:
                    if last_ty in last_pr_constraints:
                        if et not in last_pr_constraints[last_ty]:
                            continue
                    step = {'property': index_pr, 'type': et}
                    et_steps = steps[:]
                    et_steps.append(step)
                    for p in _continue(root, et_steps, index=index + 1, last_pr=index_pr, last_ty=et):
                        yield p

    for p in _continue(rt):
        yield p


def _tp_to_context(tp):
    def node(elm):
        if isinstance(elm, Variable):
            return BNode(elm.toPython())
        else:
            return elm

    return node(tp.s), node(tp.p), node(tp.o)


def _subject_transform(x):
    if isinstance(x, BNode):
        return Variable(x)
    return x


def _get_tp_paths(fountain, agp, force_seed=None):
    tp_paths = {}
    tp_hints = {}
    tp_cycles = {}
    type_dicts = {}

    wire = agp.wire
    roots = agp.roots
    graph = agp.graph

    context_force_seeds = {}

    try:
        map(lambda c: _validate_agp_context(fountain, c), graph.contexts())
    except TypeError:
        return tp_paths, tp_hints, tp_cycles

    agp_paths = {}
    bgp_paths = {}
    seed_paths = {}
    str_roots = map(lambda x: str(x), roots)
    for c in graph.contexts():
        root_tps = filter(lambda (s, pr, o): str(s).replace('?', '') in str_roots, c.triples((None, None, None)))
        root_predicates = map(lambda x: graph.qname(x[1]), root_tps)

        root_types = _root_types(fountain, root_tps, c)
        root_types = {_subject_transform(tp[0]): types for tp, types in root_types.items()}

        for tp in agp:
            if tp not in tp_hints:
                tp_hints[tp] = {}

            if isinstance(tp.s, URIRef) and tp.s in roots:
                if c not in context_force_seeds:
                    context_force_seeds[c] = []
                if (tp.s, root_types[tp.s]) not in context_force_seeds[c]:
                    for t in root_types[tp.s]:
                        context_force_seeds[c].append((tp.s, t))

            for root in roots:
                if (root, tp) not in agp_paths:
                    agp_paths[(root, tp)] = []

                if root == tp.s:
                    path = [] if (tp.p == RDF.type and isinstance(tp.o, URIRef)) else [c.qname(tp.p)]
                    agp_paths[(root, tp)].append(path)
                else:
                    try:
                        wire_paths = list(nx.all_simple_paths(wire, root, tp.s))
                    except nx.NetworkXNoPath:
                        pass
                    else:
                        for w_path in wire_paths:
                            path = []
                            source = None
                            for step in w_path:
                                if source is not None:
                                    link = graph.qname(wire.get_edge_data(source, step)['link'])
                                    # if link != 'rdf:type':
                                    path.append(link)
                                source = step
                            if tp.p != RDF.type or isinstance(tp.o, Variable):
                                path.append(c.qname(tp.p))
                            agp_paths[(root, tp)].append(path)

                if not agp_paths[(root, tp)]:
                    del agp_paths[(root, tp)]

        if not force_seed:
            force_seed = context_force_seeds.get(c, [])

        for (root, tp), paths in agp_paths.items():
            r_types = root_types.get(root, [])
            for rt in r_types:
                if (root, rt, tp) not in bgp_paths:
                    bgp_paths[(root, rt, tp)] = []

                rt_seed_paths = fountain.get_paths(rt, force_seed=force_seed)
                rt_paths = rt_seed_paths['paths']
                if not rt_paths:
                    continue

                rt_paths = filter(
                    lambda x: not x['steps'] or x['steps'][-1].get('property', None) not in root_predicates,
                    rt_paths)

                if not rt_paths:
                    continue

                rt_seed_paths['paths'] = rt_paths
                if rt not in seed_paths:
                    seed_paths[rt] = rt_seed_paths
                else:
                    for c in rt_seed_paths['all-cycles']:
                        if c not in seed_paths[rt]['all-cycles']:
                            seed_paths[rt]['all-cycles'].append(c)
                    for p in rt_paths:
                        if p not in seed_paths[rt]['paths']:
                            seed_paths[rt]['paths'].append(p)

                for path in paths:
                    for r_tp_path in _build_root_paths(fountain, path, rt):
                        bgp_paths[(root, rt, tp)].append(r_tp_path)

    for (root, rt, tp), paths in bgp_paths.items():
        if tp not in tp_paths:
            tp_cycles[tp] = []
            tp_paths[tp] = []
            tp_hints[tp] = {}

        if tp.p == RDF.type and isinstance(tp.o, URIRef) and tp.o not in type_dicts:
            type_dicts[tp.o] = fountain.get_type(graph.qname(tp.o))
            tp_hints[tp]['check'] = tp_hints[tp].get('check', False) or False

        if rt in seed_paths:
            for c in seed_paths[rt]['all-cycles']:
                if c not in tp_cycles[tp]:
                    tp_cycles[tp].append(c)

            for rt_path in seed_paths[rt]['paths']:
                for path in paths:
                    copy_rt_path = copy.deepcopy(rt_path)
                    copy_rt_path['steps'].extend(path)

                    if copy_rt_path not in tp_paths[tp]:
                        tp_paths[tp].append(copy_rt_path)
                        if tp.p == RDF.type and isinstance(tp.o, URIRef) and not tp_hints[tp]['check']:
                            rt_path_steps = rt_path['steps']
                            if rt_path_steps:
                                q_type = graph.qname(tp.o)
                                last_step = rt_path_steps[-1]
                                last_pr_dict = fountain.get_property(last_step['property'])
                                if last_step['type'] in last_pr_dict['constraints']:
                                    pr_range = last_pr_dict['constraints'][last_step['type']]
                                else:
                                    pr_range = last_pr_dict['range']
                                check = not all(map(lambda x: x == q_type or x in type_dicts[tp.o]['sub'], pr_range))
                            else:
                                check = True
                            copy_rt_path['check'] = check

    return tp_paths, tp_hints, tp_cycles


class Plan(object):
    def __init__(self, fountain, agp, force_seed=None):
        # type: (AbstractFountain, AGP) -> Plan
        self.__fountain = fountain
        log.debug('Agora Graph Pattern:\n{}'.format(agp.graph.serialize(format='turtle')))

        try:
            search, hints, cycles = _get_tp_paths(fountain, agp, force_seed=force_seed)
            self.__plan = {
                "plan": [
                    {"context": agp.get_tp_context(_tp_to_context(tp)), "pattern": _tp_to_context(tp), "paths": paths,
                     "hints": hints[tp],
                     "cycles": cycles[tp]}
                    for tp, paths in search.items()], "prefixes": agp.prefixes}

            self.__g_plan = graph_plan(self.__plan, self.__fountain, agp)
        except TypeError, e:
            traceback.print_exc()
            raise NameError(e.message)
        except Exception, e:
            traceback.print_exc()
            raise e

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
        # type: () -> AbstractFountain
        raise NotImplementedError

    def make_plan(self, *tps, **kwargs):
        # type: (set) -> Graph
        raise NotImplementedError


class Planner(AbstractPlanner):
    def __init__(self, fountain):
        # type: (AbstractFountain) -> Planner
        self.__fountain = fountain

    @property
    def fountain(self):
        return self.__fountain

    def make_plan(self, agp, force_seed=None):
        # type: (AGP) -> Graph
        plan = Plan(self.__fountain, agp, force_seed=force_seed)
        return plan.graph
