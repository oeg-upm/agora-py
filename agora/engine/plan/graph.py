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
import hashlib
import logging

from rdflib import ConjunctiveGraph, URIRef, BNode, RDF, Literal, Variable
from rdflib.namespace import Namespace, XSD, RDFS, OWL

__author__ = 'Fernando Serena'

AGORA = Namespace('http://agora.org#')

log = logging.getLogger('agora.engine.plan.graph')


def __extend_uri(prefixes, short):
    (prefix, u) = short.split(':')
    try:
        return URIRef(prefixes[prefix] + u)
    except KeyError:
        return short


def _type_subtree(fountain, t):
    return set([t] + fountain.get_type(t).get('sub'))


def graph_plan(plan, fountain, agp):
    plan_graph = ConjunctiveGraph()
    plan_graph.bind('agora', AGORA)
    prefixes = plan.get('prefixes')
    ef_plan = plan.get('plan')
    tree_lengths = {}
    s_trees = set([])
    patterns = {}
    described_cycles = {}

    for (prefix, u) in prefixes.items():
        plan_graph.bind(prefix, u)

    tree_graph = plan_graph.get_context('trees')

    def get_pattern_node(p):
        if p not in patterns:
            patterns[p] = BNode('tp_{}'.format(len(patterns)))
        return patterns[p]

    def inc_tree_length(tree, l):
        if tree not in tree_lengths:
            tree_lengths[tree] = 0
        tree_lengths[tree] += l

    def add_variable(p_node, vid, subject=True):
        sub_node = BNode(str(vid).replace('?', 'var_'))
        if subject:
            plan_graph.add((p_node, AGORA.subject, sub_node))
        else:
            plan_graph.add((p_node, AGORA.object, sub_node))
        plan_graph.set((sub_node, RDF.type, AGORA.Variable))
        plan_graph.set((sub_node, RDFS.label, Literal(str(vid), datatype=XSD.string)))

    def describe_cycle(cycle_id, cg):
        c_node = BNode('cycle{}'.format(cycle_id))
        cg = cg.get_context(c_node)
        cg.add((c_node, RDF.type, AGORA.Cycle))
        previous_node = c_node
        c_steps = cycles[cycle_id]
        cycle_type = c_steps[0].get('type')
        for et in _type_subtree(fountain, cycle_type):
            cg.add((c_node, AGORA.expectedType, __extend_uri(prefixes, et)))
        for j, step in enumerate(c_steps):
            prop = step.get('property')
            b_node = BNode(previous_node.n3() + '/' + prop)
            cg.add((b_node, AGORA.onProperty, __extend_uri(prefixes, prop)))
            c_expected_type = step.get('type')
            cg.add((b_node, AGORA.expectedType, __extend_uri(prefixes, c_expected_type)))
            cg.add((previous_node, AGORA.next, b_node))
            previous_node = b_node
        return c_node

    def include_path(elm, p_seeds, p_steps, cycles, check):
        m = hashlib.md5()
        for s in p_seeds:
            m.update(s)
        elm_uri = __extend_uri(prefixes, elm)
        b_tree = BNode(m.digest().encode('base64').strip())
        s_trees.add(b_tree)
        tree_graph.set((b_tree, RDF.type, AGORA.SearchTree))
        tree_graph.add((b_tree, AGORA.fromType, elm_uri))

        for seed in p_seeds:
            tree_graph.add((b_tree, AGORA.hasSeed, URIRef(seed)))

        for cycle_id in filter(lambda x: x not in described_cycles.keys(), cycles):
            c_node = describe_cycle(cycle_id, plan_graph)
            described_cycles[cycle_id] = c_node
            plan_graph.get_context(c_node).add((b_tree, AGORA.goesThroughCycle, c_node))

        previous_node = b_tree
        inc_tree_length(b_tree, len(p_steps))

        for j, step in enumerate(p_steps):
            prop = step.get('property')
            if j < len(p_steps) - 1 or (pattern[1] == RDF.type and isinstance(pattern[2], URIRef)):
                b_node = BNode(previous_node.n3() + '/' + prop)
                tree_graph.add((b_node, AGORA.onProperty, __extend_uri(prefixes, prop)))
            else:
                b_node = BNode(previous_node.n3() + '/#end')
            tree_graph.add((b_node, AGORA.expectedType, __extend_uri(prefixes, step.get('type'))))
            tree_graph.add((previous_node, AGORA.next, b_node))
            previous_node = b_node

        p_node = get_pattern_node(pattern)
        if pattern[1] == RDF.type and isinstance(pattern[2], URIRef):
            b_id = previous_node.n3()
            b_id += '/#end'

            b_node = BNode(b_id)
            tree_graph.add((b_node, AGORA.expectedType, pattern[2]))
            tree_graph.add((previous_node, AGORA.next, b_node))
            tree_graph.add((b_node, AGORA.byPattern, p_node))
            if check:
                tree_graph.add((b_node, AGORA.checkType, Literal(check)))
        else:
            tree_graph.add((previous_node, AGORA.byPattern, p_node))

    for i, tp_plan in enumerate(ef_plan):
        paths = tp_plan.get('paths')
        pattern = tp_plan.get('pattern')
        hints = tp_plan.get('hints')
        cycles = {}
        for c in tp_plan.get('cycles'):
            cid = str(c['cycle'])
            c_steps = c['steps']
            cycles[cid] = c_steps
            if len(c_steps) > 1:
                cycles[cid + 'r'] = list(reversed(c_steps))
        context = BNode('space_{}'.format(tp_plan.get('context')))

        for path in paths:
            steps = path.get('steps')
            seeds = path.get('seeds')
            check = path.get('check', None)
            ty = None
            if not len(steps) and len(seeds):
                ty = pattern[2]
            elif len(steps):
                ty = steps[0].get('type')
            if ty:
                include_path(ty, seeds, steps, cycles, check)

        for t in s_trees:
            tree_graph.set((t, AGORA.length, Literal(tree_lengths.get(t, 0), datatype=XSD.integer)))

        pattern_node = get_pattern_node(pattern)
        plan_graph.add((context, AGORA.definedBy, pattern_node))
        plan_graph.set((context, RDF.type, AGORA.SearchSpace))
        plan_graph.add((pattern_node, RDF.type, AGORA.TriplePattern))
        plan_graph.add((pattern_node, RDFS.label, Literal(pattern_node.toPython())))
        (sub, pred, obj) = pattern

        if isinstance(sub, BNode):
            add_variable(pattern_node, str(sub))
        elif isinstance(sub, URIRef):
            plan_graph.add((pattern_node, AGORA.subject, sub))

        if isinstance(obj, BNode):
            add_variable(pattern_node, str(obj), subject=False)
        elif isinstance(obj, Literal):
            node = BNode(str(obj).replace(' ', '').replace(':', ''))
            plan_graph.add((pattern_node, AGORA.object, node))
            plan_graph.set((node, RDF.type, AGORA.Literal))
            plan_graph.set((node, AGORA.value, obj))
        else:
            plan_graph.add((pattern_node, AGORA.object, obj))

        plan_graph.add((pattern_node, AGORA.predicate, pred))
        if pred == RDF.type:
            if 'check' in hints:
                plan_graph.add((pattern_node, AGORA.checkType, Literal(hints['check'], datatype=XSD.boolean)))

    c_roots = {}
    for c_id, c_node in described_cycles.items():
        c_root_types = set({})
        for crt in plan_graph.objects(c_node, AGORA.expectedType):
            crt_qname = plan_graph.qname(crt)
            c_root_types.update(_type_subtree(fountain, crt_qname))
        c_roots[c_id] = c_root_types

    expected_res = tree_graph.query("""SELECT DISTINCT ?n WHERE {
                                      ?n agora:expectedType ?type
                                   }""")
    node_types = {}
    roots = set()
    for r in agp.roots:
        str_r = str(r)
        if isinstance(r, Variable):
            str_r = '?' + str_r
        roots.add(str_r)
    for res in expected_res:
        to_be_extended = True
        # type_expansion = False
        near_patterns = set(tree_graph.objects(res.n, AGORA.byPattern))
        for prev in tree_graph.subjects(AGORA.next, res.n):
            for sib_node in tree_graph.objects(prev, AGORA.next):
                if sib_node != res.n:
                    near_patterns.update(set(tree_graph.objects(sib_node, AGORA.byPattern)))
        expected_types = list(tree_graph.objects(res.n, AGORA.expectedType))

        q_expected_types = set(map(lambda x: tree_graph.qname(x), expected_types))
        q_expected_types = filter(
            lambda x: not set.intersection(set(fountain.get_type(x)['super']), q_expected_types), q_expected_types)
        type_hierarchy = len(q_expected_types) == 1
        tree_graph.add((res.n, AGORA.typeHierarchy, Literal(type_hierarchy)))

        for p_node in near_patterns:
            p_pred = list(plan_graph.objects(p_node, AGORA.predicate)).pop()
            if p_pred == RDF.type:
                p_type = list(plan_graph.objects(p_node, AGORA.object)).pop()
                if isinstance(p_type, URIRef):
                    # type_expansion = True
                    for et in expected_types:
                        # tree_graph.remove((res.n, AGORA.expectedType, et))
                        if et == p_type:
                            q_expected_types = _type_subtree(fountain, tree_graph.qname(et))
                            for et_q in q_expected_types:
                                tree_graph.add((res.n, AGORA.expectedType, __extend_uri(prefixes, et_q)))
            p_subject = list(plan_graph.objects(p_node, AGORA.subject)).pop()
            if not isinstance(p_subject, URIRef):
                subject_str = list(plan_graph.objects(p_subject, RDFS.label)).pop().toPython()
            else:
                subject_str = str(p_subject)
            if subject_str not in roots:
                to_be_extended = False

        if to_be_extended:
            node_types[res.n] = set([tree_graph.qname(t) for t in tree_graph.objects(res.n, AGORA.expectedType)])

                # if not type_expansion:
                #     tree_graph.remove((res.n, AGORA.expectedType, None))
                #     q_expected_types = set(map(lambda x: tree_graph.qname(x), expected_types))
                #     q_expected_types = filter(
                #         lambda x: not set.intersection(set(fountain.get_type(x)['sub']), q_expected_types), q_expected_types)
                #     for et_q in q_expected_types:
                #         tree_graph.add((res.n, AGORA.expectedType, __extend_uri(prefixes, et_q)))

    for c_id, root_types in c_roots.items():
        found_extension = False
        for n, expected in node_types.items():
            if set.intersection(expected, set(root_types)):
                tree_graph.add((n, AGORA.isCycleStartOf, described_cycles[c_id]))
                found_extension = True

        if not found_extension:
            plan_graph.remove_context(plan_graph.get_context(described_cycles[c_id]))

    for t in s_trees:
        tree_graph.set((t, AGORA.length, Literal(tree_lengths.get(t, 0), datatype=XSD.integer)))
        from_types = set([plan_graph.qname(x) for x in plan_graph.objects(t, AGORA.fromType)])
        def_from_types = filter(lambda x: not set.intersection(set(fountain.get_type(x)['sub']), from_types),
                                from_types)
        for dft in def_from_types:
            tree_graph.set((t, AGORA.fromType, __extend_uri(prefixes, dft)))

    predicates = set(plan_graph.objects(predicate=AGORA.onProperty))
    predicates.update(set(plan_graph.objects(predicate=AGORA.predicate)))
    for p in predicates:
        qp = plan_graph.qname(p)
        try:
            p_dict = fountain.get_property(qp)
            inverse = p_dict.get('inverse', [])
            for ip in inverse:
                ext_ip = __extend_uri(prefixes, ip)
                plan_graph.add((p, OWL.inverseOf, ext_ip))
                plan_graph.add((ext_ip, OWL.inverseOf, p))
        except:
            pass

    for res in plan_graph.query("""SELECT ?tree ?sub ?nxt WHERE {
                           ?tree a agora:SearchTree ;                              
                                 agora:next ?nxt .
                           ?nxt agora:byPattern [
                                   agora:subject ?sub 
                                ]                    
                        }"""):
        if isinstance(res.sub, URIRef):
            plan_graph.set((res.tree, AGORA.hasSeed, res.sub))
            plan_graph.remove((res.nxt, AGORA.isCycleStartOf, None))

    return plan_graph
