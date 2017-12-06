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
import traceback
from datetime import datetime as dt

import redis
from concurrent.futures import wait

from agora.engine.fountain.schema import Schema
from agora.engine.utils.cache import cached

__author__ = 'Fernando Serena'

log = logging.getLogger('agora.engine.fountain.index')


def __get_by_pattern(r, pattern, func):
    # type: (redis.StrictRedis, str, callable) -> list

    def get_all():
        for k in pkeys:
            yield func(k)

    pkeys = r.keys(pattern)
    return list(get_all())


def __remove_from_sets(r, values, *args):
    # type: (redis.StrictRedis, iter, iter) -> None

    for pattern in args:
        keys = r.keys(pattern)
        for dk in keys:
            key_parts = dk.split(':')
            ef_values = values
            if len(key_parts) > 1:
                ef_values = filter(lambda x: x.split(':')[0] != key_parts[1], values)
            if len(ef_values):
                r.srem(dk, *ef_values)


def __get_vocab_set(r, pattern, vid=None):
    # type: (redis.StrictRedis, str, str) -> list

    if vid is not None:
        pattern = pattern.replace(':*:', ':%s:' % vid)
    all_sets = map(lambda x: r.smembers(x), r.keys(pattern))
    return list(reduce(set.union, all_sets, set([])))


def __extract_type(schema, r, t, vid):
    # type: (Schema, redis.StrictRedis, str, str) -> None

    log.debug('Extracting type {} from the {} vocabulary...'.format(t, vid))
    with r.pipeline() as pipe:
        pipe.multi()
        pipe.sadd('vocabs:{}:types'.format(vid), t)
        for s in schema.get_supertypes(t):
            pipe.sadd('vocabs:{}:types:{}:super'.format(vid, t), s)
        for s in schema.get_subtypes(t):
            pipe.sadd('vocabs:{}:types:{}:sub'.format(vid, t), s)
        for s in schema.get_type_properties(t):
            pipe.sadd('vocabs:{}:types:{}:props'.format(vid, t), s)
        for s in schema.get_type_specific_properties(t):
            pipe.sadd('vocabs:{}:types:{}:s-props'.format(vid, t), s)
        for s in schema.get_type_references(t):
            pipe.sadd('vocabs:{}:types:{}:refs'.format(vid, t), s)
        for s in schema.get_type_specific_references(t):
            pipe.sadd('vocabs:{}:types:{}:s-refs'.format(vid, t), s)
        pipe.execute()


def __extract_property(schema, r, p, vid):
    # type: (Schema, redis.StrictRedis, str, str) -> None

    def p_type():
        if schema.is_object_property(p):
            return 'object'
        else:
            return 'data'

    try:
        log.debug('Extracting property {} from the {} vocabulary...'.format(p, vid))
        with r.pipeline() as pipe:
            pipe.multi()
            pipe.sadd('vocabs:{}:properties'.format(vid), p)
            pipe.hset('vocabs:{}:properties:{}'.format(vid, p), 'uri', p)
            for dc in list(schema.get_property_domain(p)):
                pipe.sadd('vocabs:{}:properties:{}:_domain'.format(vid, p), dc)
            for dc in list(schema.get_property_range(p)):
                pipe.sadd('vocabs:{}:properties:{}:_range'.format(vid, p), dc)
            for dc in list(schema.get_property_inverses(p)):
                pipe.sadd('vocabs:{}:properties:{}:_inverse'.format(vid, p), dc)

            for cn in list(schema.get_property_constraints(p)):
                pipe.sadd('vocabs:{}:properties:{}:_cons'.format(vid, p), cn)

            pipe.set('vocabs:{}:properties:{}:_type'.format(vid, p), p_type())
            pipe.execute()
    except TypeError, e:
        traceback.print_exc()


def __extract_types(schema, r, vid, trace=None):
    # type: (Schema, redis.StrictRedis, str, set) -> iter
    types = schema.get_types(context=vid)

    other_vocabs = filter(lambda x: x != vid, schema.contexts)
    dependent_types = set([])
    dependent_props = set([])
    for ovid in other_vocabs:
        o_types = [t for t in _get_types(r, ovid) if t not in types]
        for oty in o_types:
            otype = _get_type(r, oty)
            if set.intersection(types, otype.get('super')) or set.intersection(types, otype.get('sub')):
                dependent_types.add((ovid, oty))
        o_props = [t for t in _get_properties(r, ovid)]
        for op in o_props:
            oprop = _get_property(r, op)
            if set.intersection(types, oprop.get('domain')) or set.intersection(types, oprop.get('range')):
                dependent_props.add((ovid, op))

    types = set.union(set([(vid, t) for t in types]), dependent_types)
    futures = []
    if trace is None:
        trace = set([])
    for v, t in types:
        if t not in trace:
            __extract_type(schema, r, t, v)
            trace.add(t)
    for v, p in dependent_props:
        if p not in trace:
            __extract_property(schema, r, p, v)
            trace.add(p)
    return types, futures


def __extract_properties(schema, r, vid, trace=None):
    # type: (Schema, redis.StrictRedis, str, set) -> iter

    properties = schema.get_properties(context=vid)

    other_vocabs = filter(lambda x: x != vid, schema.contexts)
    dependent_types = set([])
    for ovid in other_vocabs:
        o_types = [t for t in _get_types(r, ovid)]
        for oty in o_types:
            o_type = _get_type(r, oty)
            if set.intersection(properties, o_type.get('refs')) or set.intersection(properties,
                                                                                    o_type.get('properties')):
                dependent_types.add((ovid, oty))

    futures = []
    if trace is None:
        trace = set([])
    for p in properties:
        if p not in trace:
            __extract_property(schema, r, p, vid)
            trace.add(p)
    for v, ty in dependent_types:
        if ty not in trace:
            __extract_type(schema, r, ty, v)
            trace.add(ty)
    return properties, futures


def _delete_vocabulary(r, vid):
    # type: (redis.StrictRedis, str) -> None

    v_types = _get_types(r, vid)
    if len(v_types):
        __remove_from_sets(r, v_types, '*:_domain', '*:_range', '*:sub', '*:super')
    v_props = _get_properties(r, vid)
    if len(v_props):
        __remove_from_sets(r, v_props, '*:refs', '*:props')

    v_keys = r.keys('vocabs:{}:*'.format(vid))
    if len(v_keys):
        r.delete(*v_keys)


def _extract_vocabulary(schema, r, vid):
    # type: (Schema, redis.StrictRedis, str) -> (iter, iter)

    log.info('Extracting vocabulary {}...'.format(vid))
    _delete_vocabulary(r, vid)
    start_time = dt.now()
    extracted = set([])
    types, t_futures = __extract_types(schema, r, vid, trace=extracted)
    properties, p_futures = __extract_properties(schema, r, vid, trace=extracted)
    wait(p_futures + t_futures)
    log.info('Done (in {}ms)'.format((dt.now() - start_time).total_seconds() * 1000))
    return types, properties


def _get_types(r, vid=None):
    # type: (redis.StrictRedis, str) -> iter

    def shared_type(t):
        return any(filter(lambda k: r.sismember(k, t), all_type_keys))

    vid_types = __get_vocab_set(r, 'vocabs:*:types', vid)

    if vid is not None:
        all_type_keys = filter(lambda k: k != 'vocabs:{}:types'.format(vid), r.keys("vocabs:*:types"))
        vid_types = filter(lambda t: not shared_type(t), vid_types)

    return vid_types


def _get_properties(r, vid=None):
    # type: (redis.StrictRedis, str) -> iter

    def shared_property(t):
        return any(filter(lambda k: r.sismember(k, t), all_prop_keys))

    vid_props = __get_vocab_set(r, 'vocabs:*:properties', vid)

    if vid is not None:
        all_prop_keys = filter(lambda k: k != 'vocabs:{}:properties'.format(vid), r.keys("vocabs:*:properties"))
        vid_props = filter(lambda t: not shared_property(t), vid_props)

    return vid_props


def _get_property(r, prop):
    # type: (redis.StrictRedis, str) -> dict

    def get_inverse_domain(ip):
        return reduce(set.union, __get_by_pattern(r, '*:properties:{}:_domain'.format(ip), r.smembers), set([]))

    def get_inverse_range(ip):
        return reduce(set.union, __get_by_pattern(r, '*:properties:{}:_range'.format(ip), r.smembers), set([]))

    all_prop_keys = r.keys('*:properties')
    if not filter(lambda k: r.sismember(k, prop), all_prop_keys):
        raise TypeError('Unknown property')

    domain = reduce(set.union, __get_by_pattern(r, '*:properties:{}:_domain'.format(prop), r.smembers), set([]))
    rang = reduce(set.union, __get_by_pattern(r, '*:properties:{}:_range'.format(prop), r.smembers), set([]))
    inv = reduce(set.union, __get_by_pattern(r, '*:properties:{}:_inverse'.format(prop), r.smembers), set([]))
    cons = reduce(set.union, __get_by_pattern(r, '*:properties:{}:_cons'.format(prop), r.smembers), set([]))
    cons = map(lambda x: eval(x), cons)
    if len(inv):
        inverse_dr = [(get_inverse_domain(i), get_inverse_range(i)) for i in inv]
        for dom, ra in inverse_dr:
            domain.update(ra)
            rang.update(dom)

    ty = __get_by_pattern(r, '*:properties:{}:_type'.format(prop), r.get)
    try:
        ty = ty.pop()
    except IndexError:
        ty = 'object'

    return {'domain': list(domain),
            'range': list(rang),
            'inverse': list(inv),
            'constraints': dict(cons),
            'type': ty}


def _is_property(r, prop):
    # type: (redis.StrictRedis, str) -> bool
    all_prop_keys = r.keys('*:properties')
    return bool(filter(lambda k: r.sismember(k, prop), all_prop_keys))


def _is_type(r, ty):
    # type: (redis.StrictRedis, str) -> bool
    all_type_keys = r.keys('*:types')
    return bool(filter(lambda k: r.sismember(k, ty), all_type_keys))


def _get_type(r, ty):
    # type: (redis.StrictRedis, str) -> dict
    all_type_keys = r.keys('*:types')
    if not filter(lambda k: r.sismember(k, ty), all_type_keys):
        raise TypeError('Unknown type: {}'.format(ty))

    super_types = reduce(set.union, __get_by_pattern(r, '*:types:{}:super'.format(ty), r.smembers), set([]))
    sub_types = reduce(set.union, __get_by_pattern(r, '*:types:{}:sub'.format(ty), r.smembers), set([]))
    type_props = reduce(set.union, __get_by_pattern(r, '*:types:{}:props'.format(ty), r.smembers), set([]))
    type_s_props = reduce(set.union, __get_by_pattern(r, '*:types:{}:s-props'.format(ty), r.smembers), set([]))
    type_refs = reduce(set.union, __get_by_pattern(r, '*:types:{}:refs'.format(ty), r.smembers), set([]))
    type_s_refs = reduce(set.union, __get_by_pattern(r, '*:types:{}:s-refs'.format(ty), r.smembers), set([]))

    return {'super': list(super_types),
            'sub': list(sub_types),
            'properties': list(type_props),
            'spec_properties': list(type_s_props),
            'refs': list(type_refs),
            'spec_refs': list(type_s_refs)}


def _cached(f):
    # type: (callable) -> callable
    def wrap(self=None, *args, **kwargs):
        return cached(self.schema.cache)(f)(self, *args, **kwargs)

    return wrap


class Index(object):
    def __init__(self):
        self.__schema = None
        self.__r = None

    @property
    def schema(self):
        # type: () -> Schema
        return self.__schema

    @schema.setter
    def schema(self, schema):
        # type: (Schema) -> None
        self.__schema = schema

    @property
    def r(self):
        # type: () -> redis.StrictRedis
        return self.__r

    @r.setter
    def r(self, r):
        # type: (redis.StrictRedis) -> None
        self.__r = r

    @property
    def types(self):
        return _get_types(self.r)

    @property
    def properties(self):
        return _get_properties(self.r)

    @_cached
    def get_type(self, t):
        # type: (str) -> dict
        return _get_type(self.r, t)

    @_cached
    def get_property(self, p):
        # type: (str) -> dict
        return _get_property(self.r, p)

    @_cached
    def is_type(self, t):
        return _is_type(self.r, t)

    @_cached
    def is_property(self, p):
        return _is_property(self.r, p)

    def index_vocabulary(self, vid):
        return _extract_vocabulary(self.__schema, self.__r, vid)

    def delete_vocabulary(self, vid):
        return _delete_vocabulary(self.__r, vid)