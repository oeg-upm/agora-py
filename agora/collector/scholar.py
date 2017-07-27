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
import hashlib
import logging
import traceback
from Queue import Empty, Full, Queue
from StringIO import StringIO
from threading import Event, Lock, Thread

import networkx as nx
import redis
from agora.collector import Collector, triplify
from agora.collector.execution import FilterTree, StopException
from agora.engine.plan.agp import TP, AGP
from agora.engine.plan.graph import AGORA
from agora.engine.utils import stopped, Singleton
from agora.engine.utils.graph import get_triple_store
from agora.engine.utils.kv import get_kv
from agora.graph import extract_tps_from_plan, extract_seed_types_from_plan
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait
from datetime import datetime, timedelta
from rdflib import ConjunctiveGraph, Graph
from rdflib import Literal, RDF, RDFS, URIRef, Variable
from rdflib.plugins.sparql.algebra import translateQuery
from rdflib.plugins.sparql.parser import Query
from rdflib.plugins.sparql.parser import expandUnicodeEscapes
from rdflib.plugins.sparql.sparql import QueryContext
from shortuuid import uuid

__author__ = 'Fernando Serena'

log = logging.getLogger('agora.collector.scholar')

tpool = ThreadPoolExecutor(max_workers=50)


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

    s, p, o = tp
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
        self.__aborted = False
        self.__tp_map = {}
        self.__seed_types = {}
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
    def seed_types(self):
        return frozenset(self.__seed_types)

    @property
    def seed_digests(self):
        return self.__seed_digests

    @property
    def updated(self):
        #     with self.lock:
        self.__updated = False if self.kv.get('{}:updated'.format(self.key)) is None else True
        return self.__updated

    @property
    def updated_ts(self):
        # with self.lock:
        upd_ts = self.kv.get('{}:updated'.format(self.key)) or 0
        upd_dt = datetime.utcfromtimestamp(upd_ts)
        return upd_dt

    @property
    def newcomer(self):
        # with self.lock:
        return self.kv.get('{}:stored'.format(self.key)) is None

    def updated_for(self, ttl):
        ttl = int(min(10000000, ttl))
        ttl = int(max(ttl, 1))
        self.__updated = ttl > 0
        updated_key = '{}:updated'.format(self.key)
        with self.kv.pipeline() as pipe:
            if self.__updated:
                pipe.set(updated_key, ttl)
                pipe.set('{}:stored'.format(self.key), True)
                pipe.expire(updated_key, int(min(10000000, ttl)))
            else:
                pipe.delete(updated_key)
            pipe.execute()
        log.info('Fragment {} will be up-to-date for {}s'.format(self.fid, ttl))

    @property
    def collecting(self):
        # with self.lock:
        return False if self.kv.get('{}:collecting'.format(self.key)) is None else True

    @property
    def aborted(self):
        return self.__aborted

    @property
    def stored(self):
        # with self.lock:
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
        except Exception, e:
            traceback.print_exc()
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
        def w_listen(ts):
            def listen(quad):
                if datetime.utcnow() > ts:
                    try:
                        listen_queue.put_nowait(quad)
                    except Full as e:
                        log.warn(e.message)
                    except Exception:
                        traceback.print_exc()

            return listen

        if self.stored:
            for c in self.__tp_map:
                for s, p, o in self.triples.get_context(str((self.fid, self.__tp_map[c]))):
                    yield self.__tp_map[c], s, p, o
        else:
            until = datetime.utcnow()
            listener = w_listen(until)
            try:
                until_ts = calendar.timegm(until.timetuple())
                listen_queue = Queue(maxsize=100)
                with self.__lock:
                    self.__observers.add(listener)

                for c, s, p, o in self.stream.get(until_ts):
                    yield self.__tp_map[c], s, p, o

                while not self.__aborted and (not self.__updated or not listen_queue.empty()):
                    try:
                        c, s, p, o = listen_queue.get(timeout=0.1)
                        yield self.__tp_map[c], s, p, o
                    except Empty:
                        pass
                    except KeyboardInterrupt:
                        stopped.set()
                    except Exception:
                        traceback.print_exc()
                        break
            finally:
                with self.__lock:
                    self.__observers.remove(listener)

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
            g = Graph()
            plan_str = p.skolemize(g).serialize(format='turtle')
            pipe.set('{}:plan'.format(self.key), plan_str)
            pipe.execute()
        self.__tp_map = extract_tps_from_plan(self.__plan)
        self.__seed_types = extract_seed_types_from_plan(self.__plan)
        self.__calculate_seed_digests()
        self.__plan_event.set()

    def __calculate_seed_digests(self):
        self.__seed_digests = {}
        for type, seeds in self.__seed_types.items():
            m = hashlib.md5()
            for seed in sorted(seeds):
                m.update(seed)
            self.__seed_digests[type] = m.digest().encode('base64').strip()

    def __notify(self, quad):
        with self.__lock:
            for observer in self.__observers:
                observer(quad)

    def populate(self, collector):
        self.collecting = True
        self.stream.clear()

        completed = False
        n_triples = 0

        try:
            collect_dict = collector.get_fragment_generator(self.agp, filters=self.filters)
            generator = collect_dict['generator']
            self.plan = collect_dict['plan']
            self.__aborted = False
        except Exception:
            self.__plan_event.set()
            self.__aborted = True
        else:
            back_id = uuid()
            pre_time = datetime.utcnow()
            try:
                while not completed:
                    c, s, p, o = generator.next()
                    tp = self.__tp_map[str(c.node)]
                    self.stream.put(str(c.node), (s, p, o))
                    self.triples.get_context(str((back_id, tp))).add((s, p, o))
                    self.__notify((str(c.node), s, p, o))
                    n_triples += 1
            except StopIteration:
                completed = True
            except StopException:
                self.__aborted = True
            except Exception:
                self.__aborted = True

        try:
            with self.lock:
                if not stopped.is_set() and completed and not self.aborted:
                    # Replace graph store and update ttl
                    for tp in self.__tp_map.values():
                        self.triples.remove_context(self.triples.get_context(str((self.fid, tp))))
                        self.triples.get_context(str((self.fid, tp))).__iadd__(
                            self.triples.get_context(str((back_id, tp))))
                        self.triples.remove_context(self.triples.get_context(str((back_id, tp))))
                    actual_ttl = collect_dict.get('ttl')()
                    if not n_triples and actual_ttl >= 10000000:
                        actual_ttl = 0
                    elapsed = (datetime.utcnow() - pre_time).total_seconds()
                    fragment_ttl = max(actual_ttl, elapsed)
                    self.updated_for(fragment_ttl)
                    log.info(
                        'Finished fragment collection: {} ({} triples), in {}s'.format(self.fid, n_triples, elapsed))
                elif self.aborted:
                    self.remove()
        except Exception, e:
            traceback.print_exc()
        finally:
            self.collecting = False

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
            filter_map = {rev_map[v]: set(map(lambda x: x.replace(v, rev_map[v]), fs)) for v, fs in filters.items() if
                          v in rev_map}
            # There may be incoming filters that do not affect the agp
            if filter_map == {v: f for v, f in self.filters.items() if v in rev_map}:
                return agp_map


class FragmentIndex(object):
    __metaclass__ = Singleton
    instances = {}
    # i_lock = Lock()
    daemon_event = Event()
    daemon_event.clear()
    daemon_th = None

    def __init__(self, id='', **kwargs):
        # type: (any, str) -> FragmentIndex
        self.__orphaned = {}
        self.__key_prefix = id
        self.__fragments_key = '{}:fragments'.format(id)
        self.lock = Lock()
        # Load fragments from kv
        self.__fragments = dict(self.__load_fragments())
        self.__id = id

    def __new__(cls, id='', force_seed=None, **kwargs):
        index = super(FragmentIndex, cls).__new__(cls)
        planner = kwargs.get('planner', None)
        index.planner = planner
        loader = kwargs.get('loader', None)
        index.loader = loader
        cache = kwargs.get('cache', None)
        index.cache = cache
        index.force_seed = force_seed
        triples = get_triple_store(**kwargs)
        kv = get_kv(**kwargs)
        index.triples = triples
        index.kv = kv

        if FragmentIndex.daemon_th is None:
            FragmentIndex.daemon_th = Thread(target=FragmentIndex._daemon)
            FragmentIndex.daemon_th.start()

        return index

    def notify(self):
        FragmentIndex.daemon_event.set()

    @property
    def id(self):
        return self.__id

    @property
    def kv(self):
        return self.__kv

    @kv.setter
    def kv(self, k):
        self.__kv = k

    @property
    def planner(self):
        return self.__planner

    @planner.setter
    def planner(self, p):
        self.__planner = p

    @property
    def cache(self):
        return self.__cache

    @cache.setter
    def cache(self, c):
        self.__cache = c

    @property
    def triples(self):
        return self.__triples

    @triples.setter
    def triples(self, t):
        self.__triples = t

    @property
    def loader(self):
        return self.__loader

    @loader.setter
    def loader(self, l):
        self.__loader = l

    @property
    def force_seed(self):
        return self.__force_seed

    @force_seed.setter
    def force_seed(self, s):
        self.__force_seed = s

    def seed_type_digest(self, ty):
        if not self.force_seed:
            return self.planner.fountain.get_seed_type_digest(ty)
        else:
            m = hashlib.md5()
            for seed in sorted(self.force_seed[ty]):
                m.update(seed)
            current_digest = m.digest().encode('base64').strip()
            return current_digest

    def __load_fragments(self):
        fids = self.kv.smembers(self.__fragments_key)
        orph_fids = self.kv.smembers('{}:orph'.format(self.__fragments_key))

        for fragment_id in fids:
            fragment = Fragment.load(self.kv, self.triples, self.__fragments_key, fragment_id,
                                     prefixes=self.planner.fountain.prefixes)

            if fragment_id in orph_fids:
                self.kv.srem(self.__fragments_key, fragment_id)
                self.kv.srem('{}:orph'.format(self.__fragments_key), fragment_id)
                if fragment is not None:
                    fragment.remove()
                continue

            if fragment is not None:
                with fragment.lock:
                    # All those fragments that were not fully collected are marked here to be orphaned
                    if not fragment.updated and not fragment.collecting:
                        self.kv.set('{}:stored'.format(fragment.key), False)
                    else:
                        yield (fragment_id, fragment)
            else:
                self.kv.srem(self.__fragments_key, fragment_id)
                self.kv.srem('{}:orph'.format(self.__fragments_key), fragment_id)

    def get(self, agp, general=False, filters=None):
        # type: (AGP, bool) -> dict
        mapping = None
        with self.lock:
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

    @property
    def fragments(self):
        return self.__fragments

    def register(self, agp, filters=None):
        # type: (AGP, dict) -> Fragment
        with self.lock:
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
        self.__orphaned[fid] = datetime.utcnow() + timedelta(seconds=1000)

    @property
    def orphaned(self):
        return self.__orphaned

    def remove(self, fid):
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

    @staticmethod
    def _daemon():
        futures = {}
        while not stopped.is_set():
            FragmentIndex.daemon_event.clear()
            for index in FragmentIndex.instances.values():
                try:
                    with index.lock:
                        for fid in index.fragments.keys()[:]:
                            fragment = index.fragments[fid]
                            with fragment.lock:
                                if fragment.aborted:
                                    index.remove(fragment.fid)
                                elif not fragment.updated and not fragment.collecting:
                                    if not fragment.newcomer:
                                        index.remove(fragment.fid)
                                    else:
                                        collector = Collector()
                                        collector.planner = index.planner
                                        collector.cache = index.cache
                                        collector.loader = index.loader
                                        collector.force_seed = index.force_seed
                                        log.info('Starting fragment collection: {}'.format(fragment.fid))
                                        try:
                                            futures[fragment.fid] = tpool.submit(fragment.populate, collector)
                                        except RuntimeError as e:
                                            traceback.print_exc()
                                            log.warn(e.message)
                                elif fragment.updated and not index.force_seed:
                                    for t, digest in fragment.seed_digests.items():
                                        t_n3 = t.n3(fragment.agp.graph.namespace_manager)
                                        current_digest = index.seed_type_digest(t_n3)
                                        if digest != current_digest:
                                            index.remove(fragment.fid)
                                            break

                    if futures:
                        log.info('Waiting for: {} collections'.format(len(futures)))
                        wait(futures.values())
                        for fragment_id, future in futures.items():
                            exception = future.exception()
                            if exception is not None:
                                log.warn(exception.message)
                                with index.lock:
                                    try:
                                        index.remove(fragment_id)
                                    except Exception, e:
                                        traceback.print_exc()
                                        break

                        futures.clear()
                except AttributeError:
                    pass

            try:
                FragmentIndex.daemon_event.wait(timeout=1)
            except Exception:
                pass


def _map_tp(tp, vars):
    s, p, o = tp
    return TP(vars.get(s, s), vars.get(p, p), vars.get(o, o))


def _apply_filter(v, resource, filters, agp_filters):
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


class Scholar(Collector):
    def __init__(self, id='', **kwargs):
        # type: () -> Scholar
        super(Scholar, self).__init__()
        self.id = id

    def __new__(cls, id='', force_seed=None, **kwargs):
        scholar = super(Scholar, cls).__new__(cls)
        index = FragmentIndex(force_seed=force_seed, id=id, **kwargs)
        scholar.index = index
        return scholar

    @property
    def index(self):
        return self.__index

    @index.setter
    def index(self, i):
        self.__index = i

    @property
    def planner(self):
        return self.index.planner

    @property
    def force_seed(self):
        return self.index.force_seed

    @force_seed.setter
    def force_seed(self, s):
        self.index.force_seed = s

    def get(self, agp, filters=None):
        return self.index.get(agp, filters=filters)

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
        if source_plan:
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

            mapped_agp = [_map_tp(tp, m_vars) for tp in agp]
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
            tp = _map_tp(c, m_vars)

            if not in_filter_path.get(tp.s, False):
                yield tp, s, p, o
            else:
                passing = True
                if _apply_filter(tp.s, s, var_filters, m_filters):
                    ft.filter(s, None, tp.s)
                    passing = False
                if _apply_filter(tp.o, o, var_filters, m_filters):
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
            for (s, o) in candidates.get(tp, set([])).copy():
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
        mapping = self.index.get(agp, general=True, filters=filters)
        if not mapping:
            # Register fragment
            fragment = self.index.register(agp, filters=filters)
            mapping = {'fragment': fragment}

        self.index.notify()
        plan = self.mapped_plan(mapping)
        if plan:
            return {'plan': plan, 'generator': self.mapped_gen(mapping, filters),
                    'prefixes': self.index.planner.fountain.prefixes.items()}
        else:
            raise TypeError('No plan for given agp: {}'.format(agp))
