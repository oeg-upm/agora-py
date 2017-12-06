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

import networkx as nx
from networkx import Graph

from agora.engine.fountain.index import Index
from agora.engine.fountain.seed import SeedManager

__author__ = 'Fernando Serena'

log = logging.getLogger('agora.engine.fountain.path')

match_elm_cycles = {}
paths_cache = {}


def _build_directed_graph(index, generic=False, graph=None):
    # type: (Index, bool, nx.DiGraph) -> Graph

    def dom_edge():
        d_cons = p_dict['constraints'].get(d, None)
        data = {'c': []}
        if d_cons is not None:
            data['c'] = d_cons
        return d, node, data

    if graph is None:
        graph = nx.DiGraph()
    else:
        graph.clear()

    paths_cache.clear()

    graph.add_nodes_from(index.types, ty='type')
    for node in index.properties:
        p_dict = index.get_property(node)
        if p_dict['type'] != 'object':
            continue

        dom = set(p_dict.get('domain'))
        dom = filter(lambda x: index.get_type(x)['spec_refs'] or node in index.get_type(x)['spec_properties'], dom)
        ran = set(p_dict.get('range'))
        try:
            ran = filter(lambda x: node in index.get_type(x)['spec_refs'] or index.get_type(x)['spec_properties'],
                         ran)
        except TypeError:
            pass
        edges = []
        edges.extend([dom_edge() for d in dom])
        if p_dict.get('type') == 'object':
            edges.extend([(node, r) for r in ran])
        graph.add_edges_from(edges)
        graph.add_node(node, ty='prop', object=p_dict.get('type') == 'object', range=ran,
                       constraints=p_dict['constraints'])

    log.debug('Known graph: {}'.format(list(graph.edges())))
    return graph


def _find_cycles(index):
    # type: (Index) -> iter

    g_graph = _build_directed_graph(index, generic=True)
    cycle_keys = index.r.keys('*cycles*')
    for ck in cycle_keys:
        index.r.delete(ck)
    g_cycles = list(nx.simple_cycles(g_graph))
    with index.r.pipeline() as pipe:
        pipe.multi()
        for i, cy in enumerate(g_cycles):
            cycle = []
            t_cycle = None
            cycle_types = set()
            for j, elm in enumerate(cy):
                if index.is_type(elm):
                    t_cycle = elm
                elif t_cycle is not None:
                    cons = g_graph.get_edge_data(t_cycle, elm)['c']
                    if cons:
                        next_type = cy[(j + 1) % len(cy)]
                        if next_type not in cons:
                            t_cycle = None
                            cycle = []
                            cycle_types.clear()
                            break

                    cycle.append({'property': elm, 'type': t_cycle})
                    cycle_types.add(t_cycle)
                    t_cycle = None
            if t_cycle is not None:
                cons = g_graph.get_edge_data(t_cycle, cy[0])['c']
                if cons:
                    next_type = cy[1]
                    if next_type not in cons:
                        t_cycle = None
                        cycle = []
                        cycle_types.clear()

                if t_cycle:
                    cycle.append({'property': cy[0], 'type': t_cycle})
                    cycle_types.add(t_cycle)
            if cycle:
                pipe.zadd('cycles', i, cycle)
            cycle_types = reduce(lambda x, y: x.union(set(index.get_type(y)['sub'] + [y])), cycle_types, set())
            for ct in cycle_types:
                pipe.sadd('cycles:{}'.format(ct), i)
        pipe.execute()
    return g_cycles


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in xrange(0, len(l), n):
        yield l[i:i + n]


def _all_simple_paths(graph, source, target):
    if (graph, source, target) not in paths_cache:
        paths_cache[(graph, source, target)] = list(nx.all_simple_paths(graph, source, target))
    return paths_cache.get((graph, source, target), [])


def _get_simple_paths(index, graph, source, target):
    if (graph, source, target) not in paths_cache:
        paths = _all_simple_paths(graph, source, target)
        source_type_dict = index.get_type(source)
        target_type_dict = index.get_type(target)
        source_type_super = source_type_dict['super']
        target_type_super = target_type_dict['super']
        for ss_ty in source_type_super:
            paths.extend(_all_simple_paths(graph, ss_ty, target))
            for ts_ty in target_type_super:
                paths.extend(_all_simple_paths(graph, source, ts_ty))
                sp = _all_simple_paths(graph, ss_ty, ts_ty)
                paths.extend(sp)
        final_paths = []
        for p in paths:
            if p not in final_paths:
                final_paths.append(p)
        if not final_paths:
            raise nx.NetworkXNoPath
        paths_cache[(graph, source, target)] = final_paths
    return paths_cache[(graph, source, target)]


def _find_seed_paths(index, sm, graph, elm, force_seed=None):
    # elm must always be a type
    if index.is_property(elm):
        targets = index.get_property(elm)['domain']
        type = False
    elif index.is_type(elm):
        targets = [elm]
        targets.extend(index.get_type(elm)['sub'])
        type = True
    else:
        raise TypeError('{}?'.format(elm))

    targets = filter(lambda t: not set.intersection(set(index.get_type(t)['super']), targets) or
                               index.get_type(t)['spec_refs'], targets)

    if force_seed:
        seed_types = map(lambda x: x[1], force_seed)
    else:
        seed_types = sm.seeds.keys()

    seed_types = filter(lambda t: not set.intersection(set(index.get_type(t)['super']), seed_types) or
                                  index.get_type(t)['spec_properties'], seed_types)

    yielded = []
    for ty in seed_types:
        for target in targets:
            try:
                ty_paths = _get_simple_paths(index, graph, ty, target)
            except nx.NetworkXNoPath:
                ty_paths = None

            if ty_paths is not None:
                if ty == target or target in index.get_type(ty)['super']:
                    path = []
                    if not type:
                        path.append({'type': ty, 'property': elm})
                    if (ty, path) not in yielded:
                        yielded.append((ty, path))
                        yield ty, path
                for path in ty_paths:
                    path = map(lambda x: {'type': x[0], 'property': x[1]}, chunks(path[:-1], 2))
                    last_step = None
                    filtering = False
                    n_iterations = len(path)
                    if type:
                        n_iterations += 1
                    for i in range(n_iterations):
                        step_type = path[i]['type'] if i < len(path) else elm
                        if last_step is not None:
                            source = graph.get_edge_data(last_step['type'], last_step['property'])
                            link_constraints = source.get('c', [])
                            if link_constraints:
                                step_types = [step_type] + index.get_type(step_type)['sub']
                                if not set.intersection(set(step_types), set(link_constraints)):
                                    filtering = True
                                    break
                        last_step = path[i] if i < len(path) else last_step
                    if not filtering:
                        if not type:
                            path.append({'type': target, 'property': elm})
                        if (ty, path) not in yielded:
                            yielded.append((ty, path))
                            yield ty, path


def _find_path(index, sm, graph, elm, force_seed=None):
    # type: (Index, SeedManager, nx.DiGraph, str, tuple) -> tuple

    def find_seeds_for(ty):
        if force_seed:
            for s in [x[0] for x in force_seed if x[1] == ty or x[1] in index.get_type(ty)['sub']]:
                yield s
        else:
            for s in sm.get_type_seeds(ty):
                yield s

    def find_property_paths(elm):
        return list(_find_seed_paths(index, sm, graph, elm, force_seed=force_seed))

    def build_seed_path(ty, path):
        type_seeds = list(find_seeds_for(ty))
        if len(type_seeds):
            return {'cycles': [], 'seeds': type_seeds, 'steps': path}

    paths = find_property_paths(elm)
    seed_paths = filter(lambda x: x is not None, map(lambda (ty, path): build_seed_path(ty, path), paths))

    cycle_ids = set([int(c) for c in index.r.smembers('cycles:{}'.format(elm))])

    for path in seed_paths:
        path['cycles'] = cycle_ids.copy()
        for step in path['steps']:
            cycles = set([int(c) for c in index.r.smembers('cycles:{}'.format(step.get('type')))])
            path['cycles'].update(set(path['cycles']).union(cycles))
            cycle_ids.update(cycles)
        path['cycles'] = list(path['cycles'])

    applying_cycles = set(cycle_ids)

    applying_cycles = {int(cid): eval(index.r.zrange('cycles', cid, cid).pop()) for cid in applying_cycles}
    return seed_paths, [{'cycle': cid, 'steps': applying_cycles[cid]} for cid in
                        applying_cycles]


class PathManager(object):
    def __init__(self):
        # type: (Index, SeedManager) -> PathManager
        self.__index = None
        self.__sm = None
        self.__pgraph = nx.DiGraph()

    @property
    def index(self):
        return self.__index

    @index.setter
    def index(self, i):
        self.__index = i
        _build_directed_graph(self.index, graph=self.__pgraph)

    @property
    def seed_manager(self):
        return self.__sm

    @seed_manager.setter
    def seed_manager(self, s):
        self.__sm = s

    def calculate(self):
        _build_directed_graph(self.__index, graph=self.__pgraph)
        _find_cycles(self.__index)

    def get_paths(self, elm, force_seed=None):
        seed_paths, all_cycles = _find_path(self.__index, self.__sm, self.__pgraph, elm, force_seed=force_seed)
        return {'paths': seed_paths, 'all-cycles': all_cycles}

    @property
    def path_graph(self):
        return self.__pgraph

    def are_connected(self, source, target):
        try:
            return bool(_get_simple_paths(self.index, self.path_graph, source, target))
        except nx.NetworkXNoPath:
            return False
