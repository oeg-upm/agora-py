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

from rdflib import ConjunctiveGraph, URIRef, BNode, RDF, Literal
from rdflib.namespace import Namespace, XSD, RDFS

__author__ = 'Fernando Serena'

AGORA = Namespace('http://agora.org#')

log = logging.getLogger('agora.engine.plan.graph')


def __extend_uri(prefixes, short):
    (prefix, u) = short.split(':')
    try:
        return URIRef(prefixes[prefix] + u)
    except KeyError:
        return short


def graph_plan(plan, fountain):
    plan_graph = ConjunctiveGraph()
    plan_graph.bind('agora', AGORA)
    prefixes = plan.get('prefixes')
    ef_plan = plan.get('plan')
    tree_lengths = {}
    s_trees = set([])
    patterns = {}

    for (prefix, u) in prefixes.items():
        plan_graph.bind(prefix, u)

    def __get_pattern_node(p):
        if p not in patterns:
            patterns[p] = BNode('tp_{}'.format(len(patterns)))
        return patterns[p]

    def __inc_tree_length(tree, l):
        if tree not in tree_lengths:
            tree_lengths[tree] = 0
        tree_lengths[tree] += l

    def __add_variable(p_node, vid, subject=True):
        sub_node = BNode(str(vid).replace('?', 'var_'))
        if subject:
            plan_graph.add((p_node, AGORA.subject, sub_node))
        else:
            plan_graph.add((p_node, AGORA.object, sub_node))
        plan_graph.set((sub_node, RDF.type, AGORA.Variable))
        plan_graph.set((sub_node, RDFS.label, Literal(str(vid), datatype=XSD.string)))

    def include_path(elm, p_seeds, p_steps):
        elm_uri = __extend_uri(prefixes, elm)
        path_g = plan_graph.get_context(elm_uri)
        b_tree = BNode(elm_uri)
        s_trees.add(b_tree)
        path_g.set((b_tree, RDF.type, AGORA.SearchTree))
        path_g.set((b_tree, AGORA.fromType, elm_uri))

        for seed in p_seeds:
            path_g.add((b_tree, AGORA.hasSeed, URIRef(seed)))

        previous_node = b_tree
        __inc_tree_length(b_tree, len(p_steps))
        for j, step in enumerate(p_steps):
            prop = step.get('property')
            b_node = BNode(previous_node.n3() + prop)
            if j < len(p_steps) - 1 or pattern[1] == RDF.type:
                path_g.add((b_node, AGORA.onProperty, __extend_uri(prefixes, prop)))
            path_g.add((b_node, AGORA.expectedType, __extend_uri(prefixes, step.get('type'))))
            path_g.add((previous_node, AGORA.next, b_node))
            previous_node = b_node

        p_node = __get_pattern_node(pattern)
        path_g.add((previous_node, AGORA.byPattern, p_node))

    for i, tp_plan in enumerate(ef_plan):
        paths = tp_plan.get('paths')
        pattern = tp_plan.get('pattern')
        hints = tp_plan.get('hints')
        context = BNode('space_{}'.format(tp_plan.get('context')))
        for path in paths:
            steps = path.get('steps')
            seeds = path.get('seeds')
            if not len(steps) and len(seeds):
                include_path(pattern[2], seeds, steps)
            elif len(steps):
                ty = steps[0].get('type')
                include_path(ty, seeds, steps)

        for t in s_trees:
            plan_graph.set((t, AGORA.length, Literal(tree_lengths.get(t, 0), datatype=XSD.integer)))

        pattern_node = __get_pattern_node(pattern)
        plan_graph.add((context, AGORA.definedBy, pattern_node))
        plan_graph.set((context, RDF.type, AGORA.SearchSpace))
        plan_graph.add((pattern_node, RDF.type, AGORA.TriplePattern))
        (sub, pred, obj) = pattern

        if isinstance(sub, BNode):
            __add_variable(pattern_node, str(sub))
        elif isinstance(sub, URIRef):
            plan_graph.add((pattern_node, AGORA.subject, sub))

        if isinstance(obj, BNode):
            __add_variable(pattern_node, str(obj), subject=False)
        elif isinstance(obj, Literal):
            node = BNode(str(obj).replace(' ', ''))
            plan_graph.add((pattern_node, AGORA.object, node))
            plan_graph.set((node, RDF.type, AGORA.Literal))
            plan_graph.set((node, AGORA.value, obj))
        else:
            plan_graph.add((pattern_node, AGORA.object, obj))

        plan_graph.add((pattern_node, AGORA.predicate, pred))
        if pred == RDF.type:
            if 'check' in hints:
                plan_graph.add((pattern_node, AGORA.checkType, Literal(hints['check'], datatype=XSD.boolean)))

        sub_expected = plan_graph.subjects(predicate=AGORA.expectedType)
        for s in sub_expected:
            expected_types = list(plan_graph.objects(s, AGORA.expectedType))
            for et in expected_types:
                plan_graph.remove((s, AGORA.expectedType, et))
            q_expected_types = [plan_graph.qname(t) for t in expected_types]
            expected_types = [d for d in expected_types if
                              not set.intersection(set(fountain.get_type(plan_graph.qname(d)).get('super')),
                                                   set(q_expected_types))]
            for et in expected_types:
                plan_graph.add((s, AGORA.expectedType, et))

    return plan_graph
