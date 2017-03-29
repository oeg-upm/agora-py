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
import multiprocessing

import networkx as nx
from agora.engine.fountain.index import Index
from agora.engine.fountain.seed import SeedManager
from concurrent.futures import wait
from concurrent.futures.thread import ThreadPoolExecutor
from datetime import datetime as dt

from agora.engine.utils.lists import subfinder

__author__ = 'Fernando Serena'

log = logging.getLogger('agora.engine.fountain.path')

match_elm_cycles = {}
th_pool = ThreadPoolExecutor(multiprocessing.cpu_count())


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

    graph.add_nodes_from(index.types, ty='type')
    for node in index.properties:
        p_dict = index.get_property(node)
        dom = set(p_dict.get('domain'))
        if generic:
            dom = filter(lambda x: not set.intersection(set(index.get_type(x)['super']), dom), dom)
        ran = set(p_dict.get('range'))
        if generic:
            try:
                ran = filter(lambda x: not set.intersection(set(index.get_type(x)['super']), ran), ran)
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


def __store_paths(index, node_paths, g_cycles):
    # type: (Index, iter, iter) -> None

    def __get_matching_cycles(_elm):
        def __find_matching_cycles():
            for j, c in enumerate(g_cycles):
                extended_elm = [_elm]
                if index.is_type(_elm):
                    extended_elm.extend(index.get_type(_elm)["super"])

                if len([c for e in extended_elm if e in c]):
                    yield j

        if _elm not in match_elm_cycles:
            mc = set(__find_matching_cycles())
            match_elm_cycles[_elm] = mc
        return match_elm_cycles[_elm]

    def __store_path(_i, _path):
        pipe.zadd('paths:{}'.format(elm), _i, _path)

    log.debug('Preparing to persist the calculated paths...{}'.format(len(node_paths)))

    with index.r.pipeline() as pipe:
        pipe.multi()
        for (elm, paths) in node_paths:
            futures = []
            for (i, path) in enumerate(paths):
                futures.append(th_pool.submit(__store_path, i, path))
            wait(futures)
            pipe.execute()

        # Store type and property cycles
        for elm in match_elm_cycles.keys():
            for c in match_elm_cycles[elm]:
                pipe.sadd('cycles:{}'.format(elm), c)
        pipe.execute()
        for t in [_ for _ in index.types if _ not in match_elm_cycles]:
            for c in __get_matching_cycles(t):
                pipe.sadd('cycles:{}'.format(t), c)
        pipe.execute()


def __find_cycles(index):
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
            for elm in cy:
                if index.is_type(elm):
                    t_cycle = elm
                elif t_cycle is not None:
                    cycle.append({'property': elm, 'type': t_cycle})
                    t_cycle = None
            if t_cycle is not None:
                cycle.append({'property': cy[0], 'type': t_cycle})
            pipe.zadd('cycles', i, cycle)
        pipe.execute()
    return g_cycles


def __contains_cycle(g):
    return bool(list(nx.simple_cycles(g)))


def __build_paths(graph, node, root, steps=None, level=0, path_graph=None, cache=None):
    # type: (nx.DiGraph, str, str, iter, int, nx.DiGraph, iter) -> iter

    paths = []
    if steps is None:
        steps = []
    if path_graph is None:
        path_graph = nx.DiGraph()
    if cache is None:
        cache = {}

    log.debug(
        '[{}][{}] building paths to {}, with root {} and {} previous steps'.format(root, level, node, root,
                                                                                   len(steps)))

    pred = set(graph.predecessors(node))
    for t in pred:
        previous_type = root if not steps else steps[-1].get('type')
        data = graph.get_edge_data(t, node)
        if data['c']:
            if previous_type not in data['c']:
                continue

        new_path_graph = path_graph.copy()
        new_path_graph.add_nodes_from([t, node])
        new_path_graph.add_edges_from([(t, node)])

        step = {'property': node, 'type': t}
        path = [step]

        new_steps = steps[:]
        new_steps.append(step)
        log.debug('[{}][{}] added a new step {} in the path to {}'.format(root, level, (t, node), node))

        any_subpath = False
        next_steps = [x for x in graph.predecessors(t)]

        for p in next_steps:
            log.debug('[{}][{}] following {} as a pred property of {}'.format(root, level, p, t))
            extended_new_path_graph = new_path_graph.copy()
            extended_new_path_graph.add_node(p)
            extended_new_path_graph.add_edges_from([(p, t)])
            if __contains_cycle(extended_new_path_graph):
                continue
            sub_paths = __build_paths(graph, p, root, new_steps[:], level=level + 1, path_graph=extended_new_path_graph,
                                      cache=cache)

            any_subpath = any_subpath or len(sub_paths)
            for sp in sub_paths:
                paths.append(path + sp)
        if (len(next_steps) and not any_subpath) or not len(next_steps):
            paths.append(path)

    log.debug(
        '[{}][{}] returning {} paths to {}, with root {} and {} previous steps'.format(root, level, len(paths),
                                                                                       node,
                                                                                       root,
                                                                                       len(steps)))
    return paths


def __calculate_node_paths(graph, paths, n, d):
    # type: (nx.DiGraph, iter, str, dict) -> None
    log.debug('[START] Calculating paths to {} with data {}'.format(n, d))
    _paths = []
    if d.get('ty') == 'type':
        for p in graph.predecessors(n):
            log.debug('Following root [{}] predecessor property {}'.format(n, p))
            _paths.extend(__build_paths(graph, p, n))
    else:
        _paths.extend(__build_paths(graph, n, n))
    log.debug('[END] {} paths for {}'.format(len(_paths), n))
    if len(_paths):
        paths[n] = _paths


def _calculate_paths(index, graph):
    # type: (Index, nx.DiGraph) -> None

    log.info('Calculating paths...')
    match_elm_cycles.clear()
    start_time = dt.now()

    _build_directed_graph(index, graph=graph)
    g_cycles = __find_cycles(index)

    node_paths = {}
    futures = []
    for node, data in graph.nodes(data=True):
        futures.append(th_pool.submit(__calculate_node_paths, graph, node_paths, node, data))
    wait(futures)

    for ty in [_ for _ in index.types if _ in node_paths]:
        for sty in [_ for _ in index.get_type(ty)['sub'] if _ in node_paths]:
            node_paths[ty].extend(node_paths[sty])

    node_paths = node_paths.items()
    __store_paths(index, node_paths, g_cycles)

    n_paths = 0
    for path_key in index.r.keys('paths:*'):
        n_paths += index.r.zcard(path_key)

    log.info('Found {} paths in {}ms'.format(n_paths,
                                             (dt.now() - start_time).total_seconds() * 1000))


def __detect_redundancies(source, steps):
    # type: (iter, iter) -> iter
    if source and source[0] in steps:
        steps_copy = steps[:]
        start_index = steps_copy.index(source[0])
        end_index = start_index + len(source)
        try:
            cand_cycle = steps_copy[start_index:end_index]
            if end_index >= len(steps_copy):
                cand_cycle.extend(steps_copy[:end_index - len(steps_copy)])
            if cand_cycle == source:
                steps_copy = steps[0:start_index - end_index + len(steps_copy)]
                if len(steps) > end_index:
                    steps_copy += steps[end_index:]
        except IndexError:
            pass
        return steps_copy
    return steps


# @cached(seeds.cache)
def _find_path(index, sm, elm):
    # type: (Index, SeedManager, str) -> tuple

    def build_seed_path_and_identify_cycles(_seeds):
        """

        :param _seeds:
        :return:
        """
        sub_steps = list(reversed(path[:step_index + 1]))
        for _step in sub_steps:
            cycle_ids.update([int(c) for c in index.r.smembers('cycles:{}'.format(_step.get('type')))])
        sub_path = {'cycles': list(cycle_ids), 'seeds': _seeds, 'steps': sub_steps}

        if sub_path not in seed_paths:
            seed_paths.append(sub_path)
        return cycle_ids

    seed_paths = []
    paths = [(int(score), eval(path)) for path, score in index.r.zrange('paths:{}'.format(elm), 0, -1, withscores=True)]

    applying_cycles = set([])
    cycle_ids = set([int(c) for c in index.r.smembers('cycles:{}'.format(elm))])

    step_index = 0
    for score, path in paths:
        for step_index, step in enumerate(path):
            ty = step.get('type')
            type_seeds = sm.get_type_seeds(ty)
            if len(type_seeds):
                seed_cycles = build_seed_path_and_identify_cycles(type_seeds)
                applying_cycles = applying_cycles.union(set(seed_cycles))

    # It only returns seeds if elm is a type and there are seeds of it
    req_type_seeds = sm.get_type_seeds(elm)
    if len(req_type_seeds):
        path = []
        seed_cycles = build_seed_path_and_identify_cycles(req_type_seeds)
        applying_cycles = applying_cycles.union(set(seed_cycles))

    filtered_seed_paths = []
    applying_cycles = [{'cycle': int(cid), 'steps': eval(index.r.zrange('cycles', cid, cid).pop())} for cid in
                       applying_cycles]

    for cycle in applying_cycles:
        cycle_id = cycle['cycle']
        cycle_steps = cycle['steps']
        for seed_path in seed_paths:
            path_steps = seed_path['steps']
            path_cycles = seed_path['cycles']
            if cycle_id in path_cycles:
                if path_steps != cycle_steps and subfinder(path_steps, cycle_steps):
                    filtered_seed_paths.append(seed_path)

    return [_ for _ in seed_paths if _ not in filtered_seed_paths], applying_cycles


class PathManager(object):
    def __init__(self, index, sm):
        # type: (Index, SeedManager) -> PathManager
        self.__index = index
        self.__sm = sm
        self.__pgraph = nx.DiGraph()

    def calculate(self):
        _calculate_paths(self.__index, self.__pgraph)

    def get_paths(self, elm):
        seed_paths, all_cycles = _find_path(self.__index, self.__sm, elm)
        return {'paths': seed_paths, 'all-cycles': all_cycles}

    @property
    def path_graph(self):
        return self.__pgraph
