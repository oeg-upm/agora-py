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
from datetime import datetime as dt

import networkx as nx
from concurrent.futures import wait
from concurrent.futures.thread import ThreadPoolExecutor
from networkx import Graph

from agora.engine.fountain.index import Index
from agora.engine.fountain.seed import SeedManager
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


def __build_paths(index, graph, node, root, node_paths, steps=None, level=0, path_graph=None, cache=None):
    # type: (Index, nx.DiGraph, str, str, iter, iter, int, nx.DiGraph, iter) -> iter

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
    pred = filter(
        lambda t: index.get_type(t).get('sub') or not set.intersection(set(index.get_type(t).get('super')), pred), pred)

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

        infer_t = [t] + index.get_type(t)['sub']
        next_steps = map(lambda x: graph.predecessors(x), infer_t)
        next_steps = set().union(*next_steps)
        for p in next_steps:
            log.debug('[{}][{}] following {} as a pred property of {}'.format(root, level, p, t))
            extended_new_path_graph = new_path_graph.copy()
            extended_new_path_graph.add_node(p)
            extended_new_path_graph.add_edges_from([(p, t)])
            if __contains_cycle(extended_new_path_graph):
                continue

            if p in node_paths:
                sub_paths = []
                print 'already calculated'
                reuse_path_graph = new_path_graph.copy()
                for reuse_path in node_paths[p]:
                    for step in reuse_path:
                        reuse_path_graph.add_edge(step['property'], step['type'])
                    if __contains_cycle(reuse_path_graph):
                        continue
                    sub_paths.append(reuse_path)
            else:
                sub_paths = __build_paths(index, graph, p, root, node_paths, steps=new_steps[:], level=level + 1,
                                          path_graph=extended_new_path_graph,
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


def __calculate_node_paths(index, graph, paths, n, d):
    # type: (Index, nx.DiGraph, iter, str, dict) -> None
    log.debug('[START] Calculating paths to {} with data {}'.format(n, d))
    _paths = []
    if d.get('ty') == 'type':
        for p in graph.predecessors(n):
            log.debug('Following root [{}] predecessor property {}'.format(n, p))
            _paths.extend(__build_paths(index, graph, p, n, paths))
    else:
        _paths.extend(__build_paths(index, graph, n, n, paths))
    log.debug('[END] {} paths for {}'.format(len(_paths), n))
    if len(_paths):
        paths[n] = _paths


def _calculate_paths(index, graph):
    # type: (Index, nx.DiGraph) -> None

    log.info('Calculating paths...')
    match_elm_cycles.clear()
    start_time = dt.now()

    _build_directed_graph(index, graph=graph, generic=True)
    g_cycles = __find_cycles(index)

    node_paths = {}
    all_nodes = dict(graph.nodes(data=True))

    in_degree_sequence = list(graph.in_degree_iter())
    degree_sequence = sorted(map(lambda (n, d): d, in_degree_sequence))
    degrees = sorted(set(degree_sequence), reverse=True)

    for d_th in degrees:
        nodes = filter(lambda (n, deg): deg == d_th, in_degree_sequence)
        for node, _ in nodes:
            data = all_nodes[node]
            __calculate_node_paths(index, graph, node_paths, node, data)

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


def _find_path(index, sm, elm, force_seed=None):
    # type: (Index, SeedManager, str) -> tuple

    def find_seeds_for(ty):
        if force_seed:
            for (s, types) in force_seed:
                if ty in types:
                    yield s
        else:
            for s in sm.get_type_seeds(ty):
                yield s

    def find_property_paths(elm, trace=None, type=False):
        if trace is None:
            trace = []

        trace.append(elm)
        paths = get_paths_from_r(elm)
        if type:
            super_tree = set(
                filter(lambda t: not set.intersection(set(index.get_type(t)['super']), [elm]), [elm]))
            elm_tree = map(lambda t: index.get_type(t)['super'], super_tree)
            elm_tree = set().union(*elm_tree)
        else:
            elm_tree = index.get_property(elm)['domain']

        for et in elm_tree:
            if not type and subfinder(trace, [elm, et]):
                break
            et_paths = find_property_paths(et, type=True, trace=trace)
            if not type:
                et_step = [{'type': et, 'property': elm}]
                for et_p in et_paths:
                    if et_p and et_step != et_p[0]:
                        ext_path = et_step + et_p
                    if ext_path not in paths:
                        paths.append(ext_path)
            else:
                for et_path in et_paths:
                    if et_path not in paths:
                        paths.append(et_path)

        if type:
            et_refs = index.get_type(elm)['refs']
            for ref in et_refs:
                if subfinder(trace, [ref, elm]):
                    break
                ref_paths = find_property_paths(ref, type=False, trace=trace)
                for ref_path in ref_paths:
                    if ref_path not in paths:
                        paths.append(ref_path)

        return paths

    def get_paths_from_r(elm):
        return [eval(path) for path in
                index.r.zrange('paths:{}'.format(elm), 0, -1)]

    def build_seed_path(_seeds):
        """

        :param _seeds:
        :return:
        """
        sub_steps = list(reversed(path[:step_index + 1]))
        sub_path = {'cycles': [], 'seeds': _seeds, 'steps': sub_steps}

        if sub_path not in seed_paths:
            seed_paths.append(sub_path)

    # def contains_cycle(elm, path, cid):
    #     cycle_steps = applying_cycles[cid]
    #     path_steps = path['steps']
    #     if path_steps and (subfinder(path_steps, cycle_steps) or subfinder(path_steps, list(reversed(cycle_steps)))):
    #         return False
    #         # return True
    #         # return elm != cycle_steps[-1]['property']
    #         # return False
    #         # return len(cycle_steps) == len(path_steps) and elm != cycle_steps[-1]['property']
    #         # if (elm != cycle_steps[-1]['property']) or (
    #         #             cycle_steps[-1]['property'] != path_steps[-1][
    #         #             'property']):
    #         #     return True
    #     return False

    def filter_paths(paths, cycles):

        yielded = []
        # back = []

        for path in paths:
            # filtered = False
            # filtered = any([contains_cycle(elm, path, cycle_id) for cycle_id in path['cycles']])
            # if filtered:
            #     path['steps'] = []
            #     if path not in back:
            #         back.append(path)
            # else:
            if path not in yielded:
                yielded.append(path)
                yield path

        # if not yielded and paths:
        #     for b in back:
        #         yield b

    seed_paths = []

    if index.is_property(elm):
        type = False
    elif index.is_type(elm):
        type = True
    else:
        raise TypeError('{}?'.format(elm))

    paths = find_property_paths(elm, type=type)

    if force_seed is None:
        force_seed = []

    cycle_ids = set([int(c) for c in index.r.smembers('cycles:{}'.format(elm))])

    step_index = 0
    for path in paths:
        for step_index, step in enumerate(path):
            ty = step.get('type')
            type_seeds = list(find_seeds_for(ty))
            if len(type_seeds):
                build_seed_path(type_seeds)

    # It only returns seeds if elm is a type and there are seeds of it

    req_type_seeds = sm.get_type_seeds(elm)
    if len(req_type_seeds):
        path = []
        build_seed_path(req_type_seeds)

    for path in seed_paths:
        for step in path['steps']:
            cycles = set([int(c) for c in index.r.smembers('cycles:{}'.format(step.get('type')))])
            path['cycles'] = list(set(path['cycles']).union(cycles))
            cycle_ids.update(cycles)

    applying_cycles = set(cycle_ids)

    applying_cycles = {int(cid): eval(index.r.zrange('cycles', cid, cid).pop()) for cid in applying_cycles}
    return list(filter_paths(seed_paths, applying_cycles)), [{'cycle': cid, 'steps': applying_cycles[cid]} for cid in
                                                             applying_cycles]


class PathManager(object):
    def __init__(self, index, sm):
        # type: (Index, SeedManager) -> PathManager
        self.__index = index
        self.__sm = sm
        self.__pgraph = nx.DiGraph()

    def calculate(self):
        _calculate_paths(self.__index, self.__pgraph)

    def get_paths(self, elm, force_seed=None):
        seed_paths, all_cycles = _find_path(self.__index, self.__sm, elm, force_seed=force_seed)
        return {'paths': seed_paths, 'all-cycles': all_cycles}

    @property
    def path_graph(self):
        return self.__pgraph
