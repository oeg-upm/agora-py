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
import calendar
import logging
import traceback
from Queue import Empty, Full
from StringIO import StringIO
from datetime import datetime, timedelta
from multiprocessing import Queue
from threading import Thread, Event, Lock

import networkx as nx
import redis
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait
from rdflib import ConjunctiveGraph, Graph
from rdflib import Literal, RDF, RDFS, URIRef, Variable
from rdflib.plugins.sparql.algebra import translateQuery
from rdflib.plugins.sparql.parser import Query
from rdflib.plugins.sparql.parser import expandUnicodeEscapes
from rdflib.plugins.sparql.sparql import QueryContext
from shortuuid import uuid

from agora.collector import Collector, triplify
from agora.collector.cache import RedisCache
from agora.collector.execution import FilterTree
from agora.engine.plan import AbstractPlanner
from agora.engine.plan.agp import TP, AGP
from agora.engine.plan.graph import AGORA
from agora.engine.utils.graph import get_triple_store
from agora.engine.utils.kv import get_kv
from agora.graph import extract_tps_from_plan

__author__ = 'Fernando Serena'

log = logging.getLogger('agora.collector.scholar')

tpool = ThreadPoolExecutor(max_workers=4)


def _remove_tp_filters(tp, filter_mapping={}, prefixes=None):
    # type: (TP, dict, dict) -> (str, dict)
    """
    Transforms a triple pattern that may contain filters to a new one with both subject and object bounded
    to variables
    :param tp: The triple pattern to be filtered
    :return: Filtered triple pattern + filter mapping
    """

    def __create_var(elm):
        if elm in filter_mapping.values():
            elm = list(filter(lambda x: filter_mapping[x] == elm, filter_mapping)).pop()
        else:
            v = Variable('v' + uuid())
            filter_mapping[v] = elm
            elm = v
        return elm

    # g = Graph()
    s, p, o = tp
    # s, p, o = TP.from_string(tp, prefixes=prefixes, graph=g)
    if isinstance(s, URIRef):
        s = __create_var(s)
    if not isinstance(o, Variable):
        o = __create_var(o)
    return TP(s, p, o)


def _generalize_agp(agp, prefixes=None):
    # Create a filtered graph pattern from the request one (general_gp)
    general_gp = AGP(prefixes=prefixes)
    filter_mapping = {}
    for new_tp in map(lambda x: _remove_tp_filters(x, filter_mapping, prefixes=prefixes), agp):
        general_gp.add(new_tp)
    return general_gp, filter_mapping


class FragmentStream(object):
    def __init__(self, store, key):
        # type: (redis.StrictRedis, str) -> None
        self.key = key
        self.store = store

    def get(self, until):
        # type: (int) -> iter
        if until is None:
            until = '+inf'
        for x in self.store.zrangebyscore(self.key, '-inf', '{}'.format(float(until))):
            yield triplify(x)

    def put(self, tp, (s, p, o), timestamp=None):
        try:
            if timestamp is None:
                timestamp = calendar.timegm(datetime.utcnow().timetuple())
            quad = (tp, s.n3(), p.n3(), o.n3())
            not_found = not bool(self.store.zscore(self.key, quad))
            if not_found:
                with self.store.pipeline() as pipe:
                    pipe.zadd(self.key, timestamp, quad)
                    pipe.execute()
            return not_found
        except Exception as e:
            log.error(e.message)

    def clear(self):
        with self.store.pipeline() as pipe:
            pipe.delete(self.key)
            pipe.execute()


class Fragment(object):
    def __init__(self, agp, kv, triples, fragments_key, fid, filters=None):
        # type: (AGP, redis.StrictRedis, ConjunctiveGraph, str, str) -> None
        self.__lock = Lock()
        self.key = '{}:{}'.format(fragments_key, fid)
        self.__agp = agp
        self.__fragments_key = fragments_key
        self.fid = fid
        self.kv = kv
        self.triples = triples
        self.__stream = FragmentStream(kv, '{}:stream'.format(self.key))
        self.__plan = None
        self.__plan_event = Event()
        self.__plan_event.clear()
        self.__updated = False
        self.__tp_map = {}
        self.__observers = set([])
        self.collecting = False
        self.__filters = filters if isinstance(filters, dict) else {}

    @property
    def lock(self):
        return self.__lock

    @property
    def stream(self):
        return self.__stream

    @property
    def agp(self):
        return self.__agp

    @property
    def filters(self):
        return self.__filters

    @property
    def updated(self):
        with self.lock:
            self.__updated = False if self.kv.get('{}:updated'.format(self.key)) is None else True
            return self.__updated

    @property
    def newcomer(self):
        with self.lock:
            return self.kv.get('{}:stored'.format(self.key)) is None

    def updated_for(self, ttl):
        ttl = int(min(100000, ttl))  # sys.maxint don't work for expire values!
        ttl = int(max(ttl, 1))
        self.__updated = ttl > 0
        updated_key = '{}:updated'.format(self.key)
        with self.kv.pipeline() as pipe:
            if self.__updated:
                pipe.set(updated_key, ttl)
                pipe.set('{}:stored'.format(self.key), True)
                pipe.expire(updated_key, int(min(100000, ttl)))
            else:
                pipe.delete(updated_key)
            pipe.execute()
        log.info('Fragment {} will be up-to-date for {}s'.format(self.fid, ttl))

    @property
    def collecting(self):
        with self.lock:
            return False if self.kv.get('{}:collecting'.format(self.key)) is None else True

    @property
    def stored(self):
        with self.lock:
            return False if self.kv.get('{}:stored'.format(self.key)) is None else True

    @collecting.setter
    def collecting(self, state):
        collecting_key = '{}:collecting'.format(self.key)
        if state:
            self.kv.set(collecting_key, state)
        else:
            self.kv.delete(collecting_key)

    @classmethod
    def load(cls, kv, triples, fragments_key, fid, prefixes=None):
        # type: (redis.StrictRedis, ConjunctiveGraph, str, str) -> Fragment
        try:
            agp = AGP(kv.smembers('{}:{}:gp'.format(fragments_key, fid)), prefixes=prefixes)
            plan_turtle = kv.get('{}:{}:plan'.format(fragments_key, fid))
            fragment = Fragment(agp, kv, triples, fragments_key, fid)
            fragment.plan = Graph().parse(StringIO(plan_turtle), format='turtle')
            sparql_filter_keys = kv.keys('{}:{}:filters:*'.format(fragments_key, fid))
            for var_filter_key in sparql_filter_keys:
                v = var_filter_key.split(':')[-1]
                fragment.filters[Variable(v)] = set(kv.smembers(var_filter_key))

            return fragment
        except Exception:
            with kv.pipeline() as pipe:
                for fragment_key in kv.keys('{}*{}*'.format(fragments_key, fid)):
                    pipe.delete(fragment_key)
                pipe.execute()

    def save(self, pipe):
        fragment_key = '{}:{}'.format(self.__fragments_key, self.fid)
        pipe.delete(fragment_key)
        pipe.sadd('{}:gp'.format(fragment_key), *self.__agp)
        for v in self.__filters.keys():
            for f in self.__filters[v]:
                pipe.sadd('{}:filters:{}'.format(fragment_key, str(v)), f)

    @property
    def generator(self):
        def listen(quad):
            try:
                listen_queue.put_nowait(quad)
            except Full:
                pass
            except Exception:
                traceback.print_exc()

        if self.stored:
            for c in self.__tp_map:
                for s, p, o in self.triples.get_context(str((self.fid, self.__tp_map[c]))):
                    yield self.__tp_map[c], s, p, o
        else:
            try:
                until = calendar.timegm(datetime.utcnow().timetuple())
                listen_queue = Queue(maxsize=100)
                self.__observers.add(listen)
                for c, s, p, o in self.stream.get(until):
                    yield self.__tp_map[c], s, p, o

                while not self.__updated or not listen_queue.empty():
                    try:
                        c, s, p, o = listen_queue.get(timeout=0.1)
                        yield self.__tp_map[c], s, p, o
                    except Empty:
                        pass
                    except Exception:
                        traceback.print_exc()
                listen_queue.close()
            finally:
                self.__observers.remove(listen)

    @property
    def plan(self):
        # type: () -> Graph
        self.__plan_event.wait()
        return self.__plan

    @plan.setter
    def plan(self, p):
        # type: (Graph) -> None
        self.__plan = p
        with self.kv.pipeline() as pipe:
            pipe.set('{}:plan'.format(self.key), p.serialize(format='turtle'))
            pipe.execute()
        self.__tp_map = extract_tps_from_plan(self.__plan)
        self.__plan_event.set()

    def __notify(self, quad):
        for observer in self.__observers:
            observer(quad)

    def populate(self, collector):
        self.collecting = True
        self.stream.clear()
        collect_dict = collector.get_fragment_generator(self.agp, filters=self.filters)
        self.plan = collect_dict['plan']
        back_id = uuid()
        n_triples = 0
        pre_time = datetime.utcnow()
        for c, s, p, o in collect_dict['generator']:
            tp = self.__tp_map[str(c.node)]
            self.stream.put(str(c.node), (s, p, o))
            self.triples.get_context(str((back_id, tp))).add((s, p, o))
            self.__notify((str(c.node), s, p, o))
            n_triples += 1
        with self.lock:  # Replace graph store and update ttl
            for tp in self.__tp_map.values():
                self.triples.remove_context(self.triples.get_context(str((self.fid, tp))))
                self.triples.get_context(str((self.fid, tp))).__iadd__(self.triples.get_context(str((back_id, tp))))
                self.triples.remove_context(self.triples.get_context(str((back_id, tp))))
            actual_ttl = collect_dict.get('ttl')()
            if not n_triples:
                actual_ttl = 0
            elapsed = (datetime.utcnow() - pre_time).total_seconds()
            fragment_ttl = max(actual_ttl, elapsed)
            self.updated_for(fragment_ttl)
            self.collecting = False

        log.info('Finished fragment collection: {} ({} triples), in {}s'.format(self.fid, n_triples, elapsed))

    def remove(self):
        # type: () -> None
        # Clear stream
        self.stream.clear()

        # Remove graph contexts
        if self.__tp_map:
            for tp in self.__tp_map.values():
                self.triples.remove_context(self.triples.get_context(str((self.fid, tp))))

        # Remove fragment keys in kv
        with self.kv.pipeline() as pipe:
            for fragment_key in self.kv.keys('{}*{}*'.format(self.__fragments_key, self.fid)):
                pipe.delete(fragment_key)
            pipe.execute()

    def mapping(self, agp, filters):
        # type: (AGP, dict) -> dict
        agp_map = self.agp.mapping(agp)
        if agp_map:
            if not self.filters:
                return agp_map
            rev_map = {v: k for k, v in agp_map.items()}
            filter_map = {rev_map[v]: set(map(lambda x: x.replace(v, rev_map[v]), fs)) for v, fs in filters.items()}
            if filter_map == self.filters:
                return agp_map


class FragmentIndex(object):
    def __init__(self, planner, key_prefix='', kv=None, triples=None):
        # type: (AbstractPlanner, str, redis.StrictRedis) -> None
        self.__orphaned = {}
        self.__key_prefix = key_prefix
        self.__fragments_key = '{}:fragments'.format(key_prefix)
        self.kv = kv if kv is not None else get_kv()
        self.triples = triples if triples is not None else get_triple_store()
        self.__planner = planner
        self.__lock = Lock()
        # Load fragments from kv
        self.__fragments = dict(self.__load_fragments())

    def __load_fragments(self):
        fids = self.kv.smembers(self.__fragments_key)
        orph_fids = self.kv.smembers('{}:orph'.format(self.__fragments_key))

        for fragment_id in fids:
            fragment = Fragment.load(self.kv, self.triples, self.__fragments_key, fragment_id,
                                     prefixes=self.__planner.fountain.prefixes)

            if fragment_id in orph_fids:
                self.kv.srem(self.__fragments_key, fragment_id)
                self.kv.srem('{}:orph'.format(self.__fragments_key), fragment_id)
                if fragment is not None:
                    fragment.remove()
                continue

            if fragment is not None:
                yield (fragment_id, fragment)

    def get(self, agp, general=False, filters=None):
        # type: (AGP, bool) -> dict
        mapping = None
        with self.__lock:
            active_fragments = filter(lambda x: x not in self.__orphaned, self.__fragments.keys())
            for fragment_id in sorted(active_fragments, key=lambda x: abs(
                            len(self.__fragments[x].filters) - len(filters)), reverse=False):
                fragment = self.__fragments[fragment_id]
                mapping = fragment.mapping(agp, filters)
                filter_mapping = {}
                if mapping:
                    break

            if general and mapping is None:
                general, filter_mapping = _generalize_agp(agp, prefixes=self.__planner.fountain.prefixes)
                for fragment_id in active_fragments:
                    mapping = self.__fragments[fragment_id].mapping(general, {})
                    if mapping:
                        break

            if mapping is not None:
                return {'fragment': fragment, 'vars': mapping, 'filters': filter_mapping}

        return None

    @property
    def fragments(self):
        return self.__fragments

    def register(self, agp, filters=None):
        # type: (AGP, dict) -> Fragment
        with self.__lock:
            fragment_id = str(uuid())
            fragment = Fragment(agp, self.kv, self.triples, self.__fragments_key, fragment_id, filters=filters)
            with self.kv.pipeline() as pipe:
                pipe.sadd(self.__fragments_key, fragment_id)
                fragment.save(pipe)
                pipe.execute()
            self.__fragments[fragment_id] = fragment

            return fragment

    def get_fragment_stream(self, fid, until=None):
        return FragmentStream(self.kv, fid).get(until)

    def add_orphaned(self, fid):
        self.__orphaned[fid] = datetime.utcnow() + timedelta(seconds=10)

    @property
    def orphaned(self):
        return self.__orphaned

    def remove(self, fid):
        with self.__lock:
            fragment = self.__fragments[fid]
            if fid not in self.__orphaned:
                log.info('Discarding fragment: {}'.format(fragment.fid))
                self.add_orphaned(fid)
                self.kv.sadd('{}:orph'.format(self.__fragments_key), fid)
            else:
                now = datetime.utcnow()
                if now > self.__orphaned[fid]:
                    log.info('Removing fragment: {}'.format(fragment.fid))
                    self.kv.srem(self.__fragments_key, fid)
                    self.kv.srem('{}:orph'.format(self.__fragments_key), fid)
                    fragment.remove()
                    del self.__fragments[fid]


class Scholar(Collector):
    def __init__(self, planner, cache=None, loader=None):
        # type: (AbstractPlanner, RedisCache, object) -> None
        super(Scholar, self).__init__(planner, cache)
        # Scholars require cache
        self.loader = loader
        kv = None
        persist_mode = False
        triples = None
        if cache is not None:
            kv = cache.r
            persist_mode = cache.persist_mode
        if persist_mode:
            triples = get_triple_store(persist_mode=persist_mode, base=cache.base_path, path='fragments')
        self.__index = FragmentIndex(planner, key_prefix='scholar', kv=kv, triples=triples)
        self.__daemon_event = Event()
        self.__daemon_event.clear()
        self.__enabled = True
        self.__daemon = Thread(target=self._daemon)
        self.__daemon.daemon = True
        self.__daemon.start()

    def _daemon(self):
        futures = {}
        while self.__enabled:
            for fragment in self.__index.fragments.values():
                if not fragment.updated and not fragment.collecting:
                    if not fragment.newcomer:
                        self.__index.remove(fragment.fid)
                    else:
                        collector = Collector(self.planner, self.cache)
                        collector.loader = self.loader
                        log.info('Starting fragment collection: {}'.format(fragment.fid))
                        try:
                            futures[fragment.fid] = tpool.submit(fragment.populate, collector)
                        except RuntimeError as e:
                            traceback.print_exc()
                            log.warn(e.message)

            if futures:
                log.info('Waiting for: {} collections'.format(len(futures)))
                wait(futures.values())
                for fragment_id, future in futures.items():
                    exception = future.exception()
                    if exception is not None:
                        log.warn(exception.message)
                        self.__index.remove(fragment_id)

                futures.clear()
            try:
                self.__daemon_event.wait(timeout=1)
                self.__daemon_event.clear()
            except Exception:
                pass

    @staticmethod
    def __map(mapping, elm):
        map_terms = mapping.get('vars', None)
        map_literals = mapping.get('filters', None)
        result = elm
        if map_terms:
            result = map_terms.get(result, result)
        if map_literals:
            result = map_literals.get(result, result)

        return result

    def mapped_plan(self, mapping):
        source_plan = mapping['fragment'].plan
        mapped_plan = Graph()
        for prefix, uri in source_plan.namespaces():
            mapped_plan.bind(prefix, uri)
        mapped_plan.__iadd__(source_plan)
        v_nodes = list(mapped_plan.subjects(RDF.type, AGORA.Variable))
        for v_node in v_nodes:
            v_source_label = list(mapped_plan.objects(v_node, RDFS.label)).pop()
            mapped_term = self.__map(mapping, Variable(v_source_label))
            if isinstance(mapped_term, Literal):
                mapped_plan.set((v_node, RDF.type, AGORA.Literal))
                mapped_plan.set((v_node, AGORA.value, Literal(mapped_term.n3())))
                mapped_plan.remove((v_node, RDFS.label, None))
            elif isinstance(mapped_term, URIRef):
                mapped_plan.remove((v_node, None, None))
                for s, p, _ in mapped_plan.triples((None, None, v_node)):
                    mapped_plan.remove((s, p, v_node))
                    mapped_plan.add((s, p, mapped_term))
            else:
                mapped_plan.set((v_node, RDFS.label, Literal(mapped_term.n3())))

        return mapped_plan

    def __map_tp(self, tp, vars):
        s, p, o = tp
        return TP(vars.get(s, s), vars.get(p, p), vars.get(o, o))

    def __apply_filter(self, v, resource, filters, agp_filters):
        if v in filters:
            for var_f in filters[v]:
                context = QueryContext()
                context[v] = resource
                passing = var_f.expr.eval(context) if hasattr(var_f.expr, 'eval') else bool(
                    resource.toPython())
                if not passing:
                    return True
        elif v in agp_filters:
            return resource != agp_filters.get(v)
        return False

    def __follow_filter(self, candidates, tp, s, o, trace=None, prev=None):
        def seek_join(tps, f):
            for t in tps:
                for (ss, oo) in filter(lambda x: f(x), candidates[t]):
                    if (ss, t.p, oo) not in trace:
                        yield (t, ss, t.p, oo)
                        trace.add((ss, t.p, oo))
                        for q in self.__follow_filter(candidates, t, ss, oo, trace, prev=tp):
                            yield q

        if trace is None:
            trace = set([])
        trace.add((s, tp.p, o))
        valid_tps = filter(lambda x: x != prev, candidates.keys())
        up_tps = filter(lambda x: x != prev and x.o == tp.s, valid_tps)
        for q in seek_join(up_tps, lambda (ss, oo): oo == s):
            yield q
        down_tps = filter(lambda x: x.s == tp.o, valid_tps)
        for q in seek_join(down_tps, lambda (ss, oo): ss == o):
            yield q
        sib_tps = filter(lambda x: x.s == tp.s and x.o != tp.o, valid_tps)
        for q in seek_join(sib_tps, lambda (ss, oo): ss == s):
            yield q

    def mapped_gen(self, mapping, filters):
        generator = mapping['fragment'].generator
        m_vars = mapping.get('vars', {})
        m_filters = mapping.get('filters', {})
        agp = mapping['fragment'].agp
        if filters == mapping['fragment'].filters:
            filters = {}

        var_filters = {}
        ft = FilterTree()
        in_filter_path = {}
        candidates = {}

        if filters:
            for v in filters:
                var_filters[v] = []
                for f in filters[v]:
                    f = 'SELECT * WHERE { FILTER (%s) }' % f
                    parse = Query.parseString(expandUnicodeEscapes(f), parseAll=True)
                    query = translateQuery(parse)
                    var_filters[v].append(query.algebra.p.p)

            mapped_agp = [self.__map_tp(tp, m_vars) for tp in agp]
            for tp in mapped_agp:
                ft.add_tp(tp)
                if tp.s in filters or tp.s in m_filters:
                    ft.add_variable(tp.s)
                if tp.o in filters or tp.o in m_filters:
                    ft.add_variable(tp.o)

            ugraph = ft.graph.to_undirected()

            for tp in mapped_agp:
                if tp.s not in in_filter_path:
                    in_filter_path[tp.s] = False
                    for v in ft.variables:
                        try:
                            nx.has_path(ugraph, tp.s, v)
                            in_filter_path[tp.s] = True
                            break
                        except:
                            pass

        for c, s, p, o in generator:
            tp = self.__map_tp(c, m_vars)

            if not in_filter_path.get(tp.s, False):
                yield tp, s, p, o
            else:
                passing = True
                if self.__apply_filter(tp.s, s, var_filters, m_filters):
                    ft.filter(s, None, tp.s)
                    passing = False
                if self.__apply_filter(tp.o, o, var_filters, m_filters):
                    ft.filter(o, None, tp.o)
                    passing = False

                if passing:
                    if tp not in candidates:
                        candidates[tp] = set([])
                    candidates[tp].add((s, o))

        tp_filter_roots = filter(lambda x: x.s in ft.variables or x.o in ft.variables, candidates.keys())
        tp_filter_roots = sorted(tp_filter_roots, key=lambda x: len(ft.graph.successors(x.o)), reverse=True)
        for tp in tp_filter_roots:
            pairs = {}
            for (s, o) in candidates.get(tp, []):
                if tp not in pairs:
                    pairs[tp] = set([])
                pairs[tp].add((s, o))
                for ftp, fs, fp, fo in self.__follow_filter(candidates, tp, s, o):
                    if ftp not in pairs:
                        pairs[ftp] = set([])
                    pairs[ftp].add((fs, fo))
                    candidates[ftp].remove((fs, fo))

            if len(candidates) != len(pairs):
                candidates = {}
                break
            else:
                candidates = pairs.copy()

        for tp, pairs in candidates.items():
            for s, o in pairs:
                yield tp, s, tp.p, o

    def get_fragment_generator(self, agp, **kwargs):
        filters = kwargs.get('filters', {})
        mapping = self.__index.get(agp, general=True, filters=filters)
        if not mapping:
            # Register fragment
            fragment = self.__index.register(agp, filters=filters)
            mapping = {'fragment': fragment}

        self.__daemon_event.set()
        return {'plan': self.mapped_plan(mapping), 'generator': self.mapped_gen(mapping, filters),
                'prefixes': self.planner.fountain.prefixes.items()}

    def __exit__(self, type, value, traceback):
        self.shutdown()

    def __enter__(self):
        pass

    def __del__(self):
        self.shutdown()

    def shutdown(self, wait=False):
        self.__enabled = False
        tpool.shutdown(wait)
        if hasattr(self.cache, 'close'):
            self.cache.close()
