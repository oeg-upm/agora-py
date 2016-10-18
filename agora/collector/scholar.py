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
from multiprocessing import Queue
from threading import Thread, Event, Lock

import networkx as nx
import re
from agora.collector import Collector, triplify
from agora.collector.cache import RedisCache
from agora.engine.plan import AbstractPlanner
from agora.engine.plan.graph import AGORA
from agora.engine.utils import tp_parts
from agora.engine.utils.graph import get_triple_store
from agora.engine.utils.kv import get_kv
from agora.graph import extract_tp_from_plan
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait
from datetime import datetime
from networkx.algorithms.isomorphism import DiGraphMatcher
from rdflib import BNode
from rdflib import ConjunctiveGraph, Graph
from rdflib import Literal
from rdflib import RDF
from rdflib import RDFS
from rdflib import URIRef
from shortuuid import uuid

__author__ = 'Fernando Serena'

log = logging.getLogger('agora.collector.scholar')

tpool = ThreadPoolExecutor(max_workers=4)


class GraphPattern(set):
    """
    An extension of the set class that represents a graph pattern, which is a set of triple patterns
    """

    def __init__(self, s=()):
        super(GraphPattern, self).__init__(s)

    @property
    def wire(self):
        # type: () -> nx.DiGraph
        """
        Creates a graph from the graph pattern
        :return: The graph (networkx)
        """
        g = nx.DiGraph()
        for tp in self:
            (s, p, o) = tuple(tp_parts(tp.strip()))
            edge_data = {'link': p}
            g.add_node(s)
            if o.startswith('?'):
                g.add_node(o)
            else:
                g.add_node(o, literal=o)
                edge_data['to'] = o
            g.add_edge(s, o, **edge_data)

        return g

    def __eq__(self, other):
        # type: (GraphPattern) ->  bool
        """
        Two graph patterns are equal if they are isomorphic**
        """
        if not isinstance(other, GraphPattern):
            return super(GraphPattern, self).__eq__(other)

        mapping = self.mapping(other)
        return mapping is not None

    def __repr__(self):
        return str(list(self))

    def mapping(self, other):
        # type: (GraphPattern) -> iter
        """
        :return: If there is any, the mapping with another graph pattern
        """
        if not isinstance(other, GraphPattern):
            return ()

        my_wire = self.wire
        others_wire = other.wire

        def __node_match(n1, n2):
            return n1 == n2

        def __edge_match(e1, e2):
            return e1 == e2

        matcher = DiGraphMatcher(my_wire, others_wire, node_match=__node_match, edge_match=__edge_match)
        mapping = list(matcher.isomorphisms_iter())
        if len(mapping) == 1:
            return mapping.pop()
        else:
            return ()


def _remove_tp_filters(tp, filter_mapping={}):
    # type: (str) -> (str, dict)
    """
    Transforms a triple pattern that may contain filters to a new one with both subject and object bounded
    to variables
    :param tp: The triple pattern to be filtered
    :return: Filtered triple pattern + filter mapping
    """

    def __create_var(elm, predicate):
        if elm in filter_mapping.values():
            elm = list(filter(lambda x: filter_mapping[x] == elm, filter_mapping)).pop()
        elif predicate(elm):
            v = '?{}'.format(uuid())
            filter_mapping[v] = elm
            elm = v
        return elm

    s, p, o = tp_parts(tp)
    s = __create_var(s, lambda x: '<' in x and '>' in x)
    o = __create_var(o, lambda x: '"' in x or ('<' in x and '>' in x))
    return '{} {} {}'.format(s, p, o)


def _generalize_agp(agp):
    # Create a filtered graph pattern from the request one (general_gp)
    general_gp = GraphPattern()
    filter_mapping = {}
    for new_tp in map(lambda x: _remove_tp_filters(x, filter_mapping), agp):
        general_gp.add(new_tp)
    return general_gp, filter_mapping


class FragmentStream(object):
    def __init__(self, store, key):
        # type: (redis.StrictRedis, str) -> FragmentStream
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
    def __init__(self, agp, kv, triples, fragments_key, fid):
        # type: (GraphPattern, redis.StrictRedis, ConjunctiveGraph, str, str) -> Fragment
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
    def updated(self):
        with self.lock:
            self.__updated = False if self.kv.get('{}:updated'.format(self.key)) is None else True
            return self.__updated

    def updated_for(self, ttl):
        ttl = int(min(100000, ttl))  # sys.maxint don't work for expire values!
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
    def load(cls, kv, triples, fragments_key, fid):
        # type: (redis.StrictRedis, ConjunctiveGraph, str, str) -> Fragment
        agp = GraphPattern(kv.smembers('{}:{}:gp'.format(fragments_key, fid)))
        plan_turtle = kv.get('{}:{}:plan'.format(fragments_key, fid))
        fragment = Fragment(agp, kv, triples, fragments_key, fid)
        fragment.plan = Graph().parse(StringIO(plan_turtle), format='turtle')
        return fragment

    def save(self, pipe):
        fragment_key = '{}:{}'.format(self.__fragments_key, self.fid)
        pipe.delete(fragment_key)
        pipe.sadd('{}:gp'.format(fragment_key), *self.__agp)

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
            with self.lock:
                for c in self.__tp_map:
                    for s, p, o in self.triples.get_context(str((self.fid, self.__tp_map[c]))):
                        yield c, s, p, o
        else:
            try:
                until = calendar.timegm(datetime.utcnow().timetuple())
                listen_queue = Queue(maxsize=100)
                self.__observers.add(listen)
                for quad in self.stream.get(until):
                    yield quad

                while not self.__updated or not listen_queue.empty():
                    try:
                        quad = listen_queue.get(timeout=0.1)
                        yield quad
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
        self.__tp_map = extract_tp_from_plan(self.__plan)
        self.__plan_event.set()

    def __notify(self, quad):
        for observer in self.__observers:
            observer(quad)

    def populate(self, collector):
        self.collecting = True
        self.stream.clear()
        collect_dict = collector.get_fragment_generator(*self.agp)
        self.plan = collect_dict['plan']
        back_id = uuid()
        n_triples = 0
        for c, s, p, o in collect_dict['generator']:
            tp = self.__tp_map[str(c)]
            self.stream.put(str(c), (s, p, o))
            self.triples.get_context(str((back_id, tp))).add((s, p, o))
            self.__notify((str(c), s, p, o))
            n_triples += 1
        with self.lock:  # Replace graph store and update ttl
            for tp in self.__tp_map.values():
                self.triples.remove_context(self.triples.get_context(str((self.fid, tp))))
                self.triples.get_context(str((self.fid, tp))).__iadd__(self.triples.get_context(str((back_id, tp))))
                self.triples.remove_context(self.triples.get_context(str((back_id, tp))))
            self.updated_for(collect_dict.get('ttl')())
            self.collecting = False
        log.info('Finished fragment collection: {} ({} triples)'.format(self.fid, n_triples))

    def remove(self):
        # type: () -> None

        # Clear stream
        self.stream.clear()

        # Remove fragment keys in kv
        with self.kv.pipeline() as pipe:
            for fragment_key in self.kv.keys('{}*{}*'.format(self.__fragments_key, self.fid)):
                pipe.delete(fragment_key)
            pipe.execute()

        # Remove graph contexts
        if self.__tp_map:
            for tp in self.__tp_map.values():
                self.triples.remove_context(self.triples.get_context(str((self.fid, tp))))


class FragmentIndex(object):
    def __init__(self, key_prefix='', kv=None, triples=None):
        # type: (str, redis.StrictRedis) -> FragmentIndex
        self.__key_prefix = key_prefix
        self.__fragments_key = '{}:fragments'.format(key_prefix)
        self.kv = kv if kv is not None else get_kv()
        self.triples = triples if triples is not None else get_triple_store()
        # Load fragments from kv
        self.__fragments = dict(self.__load_fragments())
        self.__lock = Lock()

    def __load_fragments(self):
        fids = self.kv.smembers(self.__fragments_key)
        for fragment_id in fids:
            fragment = Fragment.load(self.kv, self.triples, self.__fragments_key, fragment_id)
            yield (fragment_id, fragment)

    def get(self, agp, general=False):
        # type: (GraphPattern, bool) -> dict
        with self.__lock:
            agp_keys = self.kv.keys('{}:*:gp'.format(self.__fragments_key))
            for agp_key in agp_keys:
                stored_agp = GraphPattern(self.kv.smembers(agp_key))
                mapping = stored_agp.mapping(agp)
                filter_mapping = {}
                if not mapping and general:
                    general, filter_mapping = _generalize_agp(agp)
                    mapping = stored_agp.mapping(general)
                if mapping:
                    fragment_id = agp_key.split(':')[-2]
                    return {'fragment': self.__fragments[fragment_id],
                            'variables': mapping, 'literals': filter_mapping}

        return None

    @property
    def fragments(self):
        return self.__fragments

    def register(self, agp, **kwargs):
        # type: (GraphPattern) -> Fragment
        with self.__lock:
            fragment_id = str(uuid())
            fragment = Fragment(agp, self.kv, self.triples, self.__fragments_key, fragment_id)
            with self.kv.pipeline() as pipe:
                pipe.sadd(self.__fragments_key, fragment_id)
                fragment.save(pipe)
                pipe.execute()
            self.__fragments[fragment_id] = fragment

            return fragment

    def get_fragment_stream(self, fid, until=None):
        return FragmentStream(self.kv, fid).get(until)

    def remove(self, fid):
        with self.__lock:
            fragment = self.__fragments[fid]
            del self.__fragments[fid]
            self.kv.srem(self.__fragments_key, fid)
            fragment.remove()


class Scholar(Collector):
    def __init__(self, planner, cache=None, loader=None):
        # type: (AbstractPlanner, RedisCache) -> Scholar
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
        self.__index = FragmentIndex(key_prefix='scholar', kv=kv, triples=triples)
        self.__daemon_event = Event()
        self.__daemon_event.clear()
        self.__enabled = True
        self.__daemon = Thread(target=self._daemon)
        self.__daemon.daemon = True
        self.__daemon.start()

    def _daemon(self):
        # type: (FragmentIndex, AbstractPlanner) -> None
        futures = {}
        while self.__enabled:
            for fragment in self.__index.fragments.values():
                # fragment.lock.acquire()
                if not fragment.updated and not fragment.collecting:
                    collector = Collector(self.planner, self.cache)
                    collector.loader = self.loader
                    log.info('Starting fragment collection: {}'.format(fragment.fid))
                    try:
                        futures[fragment.fid] = tpool.submit(fragment.populate, collector)
                    except RuntimeError as e:
                        log.warn(e.message)
                    # fragment.lock.release()

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
        map_vars = mapping.get('variables', None)
        map_literals = mapping.get('literals', None)
        result = str(elm)
        if map_vars is not None:
            result = map_vars.get(result, result)
        if map_literals is not None:
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
            new_value = Literal(self.__map(mapping, v_source_label))
            if not new_value.startswith('?'):
                mapped_plan.set((v_node, RDF.type, AGORA.Literal))
                mapped_plan.set((v_node, AGORA.value, new_value))
                mapped_plan.remove((v_node, RDFS.label, None))
            else:
                mapped_plan.set((v_node, RDFS.label, new_value))

        l_nodes = list(mapped_plan.subjects(RDF.type, AGORA.Literal))
        for l_node in l_nodes:
            l_source_label = list(mapped_plan.objects(l_node, AGORA.value)).pop()
            mapped_plan.set((l_node, AGORA.value, Literal(self.__map(mapping, l_source_label))))
        return mapped_plan

    def mapped_gen(self, mapping):
        generator = mapping['fragment'].generator
        for c, s, p, o in generator:
            yield c, s, p, o

    def get_fragment_generator(self, *tps, **kwargs):
        agp = GraphPattern(tps)

        mapping = self.__index.get(agp, general=True)
        if not mapping:
            # Register fragment
            fragment = self.__index.register(agp, **kwargs)
            mapping = {'fragment': fragment}

        self.__daemon_event.set()
        return {'plan': self.mapped_plan(mapping), 'generator': self.mapped_gen(mapping),
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
