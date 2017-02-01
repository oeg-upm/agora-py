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
import base64
import traceback

import networkx as nx
import requests
import shortuuid
from jsonpath_rw import parse
from pyld import jsonld
from rdflib import ConjunctiveGraph
from rdflib import Graph
from rdflib.term import Literal, URIRef, BNode

from agora.collector.http import extract_ttl
from agora.collector.wrapper import ResourceWrapper
from agora.engine.fountain import AbstractFountain
from agora.engine.plan.agp import extend_uri
from agora.ted.evaluate import evaluate
from agora.ted.ns import TED_NS
from agora.ted.utils import encode_rdict

__author__ = 'Fernando Serena'


def get_ns(fountain):
    g = Graph()
    prefixes = fountain.prefixes
    for prefix, ns in prefixes.items():
        g.bind(prefix, ns)
    return g.namespace_manager


def apply_mappings(data, mappings, ns):
    def value_mapping(k, data, mapping, ns):
        if k in mapping and mapping[k].transform is not None:
            return mapping[k].transform.attach(data)
        else:
            if isinstance(data, dict) or isinstance(data, list):
                return apply_mapping(data, mapping, ns)
            else:
                return data

    def apply_mapping(data, mapping, ns):
        if isinstance(data, dict):
            return {apply_mapping(k, mapping, ns): value_mapping(k, data[k], mapping, ns) for k in data}
        elif isinstance(data, list):
            return [apply_mapping(elm, mapping, ns) for elm in data]

        if data in mapping:
            return mapping[data].predicate.n3(ns)
        return data

    m_dict = {m.key: m for m in mappings}
    if not isinstance(data, dict):
        data = {m.key: data for m in mappings}
    mapped = apply_mapping(data, m_dict, ns)
    return mapped


def ld_triples(ld, g=None):
    bid_map = {}

    def parse_term(term):
        if term['type'] == 'IRI':
            return URIRef(term['value'])
        elif term['type'] == 'literal':
            datatype = term.get('datatype', None)
            return Literal(term['value'], datatype=URIRef(datatype))
        else:
            bid = term['value'].split(':')[1]
            if bid not in bid_map:
                bid_map[bid] = shortuuid.uuid()
            return BNode(bid_map[bid])

    if g is None:
        g = Graph()
    norm = jsonld.normalize(ld)
    def_graph = norm.get('@default', [])
    for triple in def_graph:
        subject = parse_term(triple['subject'])
        predicate = parse_term(triple['predicate'])
        object = parse_term(triple['object'])
        g.add((subject, predicate, object))

    return g


class Gateway(object):
    def __init__(self, ted, fountain, server_name='gateway', url_scheme='http', server_port=None, path=''):
        # type: (TED) -> None
        self.__ted = ted
        self.__fountain = fountain
        self.__ns = get_ns(self.__fountain)
        self.__seeds = set([])
        self.__wrapper = ResourceWrapper(server_name=server_name, url_scheme=url_scheme, server_port=server_port,
                                         path=path)
        self.__rdict = {t.id: t for t in ted.ecosystem.things}
        self.__ndict = {t.node: t.id for t in ted.ecosystem.things}

        self.__wrapper.intercept('{}/<tid>'.format(path))(self.describe_resource)
        self.__wrapper.intercept('{}/<tid>/<b64>'.format(path))(self.describe_resource)
        self.__network = nx.DiGraph()

        for tid, resource in self.__rdict.items():
            self.__network.add_node(resource.id)
            if resource in ted.ecosystem.roots:
                uri = URIRef(self.url_for(tid=tid)) if not isinstance(resource.node, URIRef) else resource.node
                for t in resource.types:
                    self.__seeds.add((uri, t.n3(self.__ns)))

            if resource.td_node is not None:
                parent_search_query = """
                SELECT DISTINCT ?parent WHERE {
                    ?td <%s> ?parent .
                    ?tr <%s> [ <%s> "%s" ] .
                    {
                      ?td <%s> [ <%s> ?tr ]
                    }
                    UNION
                    {
                      ?td <%s> ?tr
                    }
                }""" % (TED_NS.describes,
                        TED_NS.valuesTransformedBy,
                        TED_NS.identifier,
                        resource.id,
                        TED_NS.hasMappingRelation, TED_NS.hasMapping,
                        TED_NS.enrichesBy
                        )

                parents = map(lambda x: self.__rdict[self.__ndict[x.parent]], resource.graph.query(parent_search_query))
                for parent in parents:
                    self.__network.add_edge(parent.id, resource.id)

        print self.__network.edges()

    @property
    def seeds(self):
        return frozenset(self.__seeds)

    @property
    def base(self):
        return self.__wrapper.base

    @property
    def host(self):
        return self.__wrapper.host

    @property
    def path(self):
        return self.__wrapper.path

    def load(self, uri, format=None):
        return self.__wrapper.load(uri)

    def compose_endpoints(self, resource):
        id = resource.id
        for base_e in resource.base:
            if base_e.href is None:
                for pred in self.__network.predecessors(id):
                    pred_thing = self.__rdict[pred]
                    for pred_e in self.compose_endpoints(pred_thing):
                        yield pred_e + base_e
            else:
                yield base_e

    def evaluate_href(self, href, **kwargs):
        for v in filter(lambda x: x in href, kwargs):
            href = href.replace(v, kwargs[v])
        return evaluate(href)

    def describe_resource(self, tid, b64=None, **kwargs):
        resource = self.__rdict[tid]
        g = ConjunctiveGraph()
        ttl = 100000
        try:
            if b64 is not None:
                b64 = b64.replace('%3D', '=')
                resource_args = eval(base64.b64decode(b64))
            else:
                resource_args = {}
            r_uri = self.url_for(tid=tid, b64=b64)
            if kwargs:
                r_uri = '{}?{}'.format(r_uri, '&'.join(['{}={}'.format(k, kwargs[k]) for k in kwargs]))
            r_uri = URIRef(r_uri)

            for s, p, o in resource.triples:
                if isinstance(o, BNode) and o in self.__ndict:
                    o = URIRef(self.url_for(tid=self.__ndict[o], b64=b64))

                if s == resource.node:
                    s = r_uri

                if not (isinstance(s, BNode) and s in self.__ndict):
                    g.add((s, p, o))

            if resource.base:
                for e in self.compose_endpoints(resource):
                    href = e.href
                    href = self.evaluate_href(href, **resource_args)
                    print 'getting {}'.format(href)
                    response = requests.get(href, headers={'Accept': e.media})
                    if response.status_code == 200:
                        data = response.json()
                        if e.path is not None:
                            jsonpath_expr = parse(e.path)
                            data = [match.value for match in jsonpath_expr.find(data)]
                            if len(data) == 1:
                                data = data.pop()
                        e_mappings = resource.endpoint_mappings(e)
                        ld = self.enrich(r_uri, apply_mappings(data, e_mappings, self.__ns), resource.types,
                                         self.__fountain, ns=self.__ns, **resource_args)
                        ld_triples(ld, g)
                        ttl = min(ttl, extract_ttl(response.headers) or ttl)

                        for enrichment in resource.enrichments:
                            obj = enrichment.object
                            if obj in self.__ndict:
                                object_resource = self.__rdict[self.__ndict[obj]]
                                rdict = {v: resource_args[v] for v in object_resource.vars if v in resource_args}
                                obj = URIRef(
                                    self.url_for(tid=self.__ndict[obj], b64=encode_rdict(rdict)))
                            g.add((r_uri, enrichment.predicate, obj))

        except Exception as e:
            print e.message
            traceback.print_exc()
        return g, {'Cache-Control': 'max-age={}'.format(ttl)}

    def url_for(self, tid, b64=None):
        if isinstance(tid, BNode) and tid in self.__ndict:
            tid = self.__ndict[tid]
        return self.__wrapper.url_for('describe_resource', tid=tid, b64=b64)

    def enrich(self, uri, data, types, fountain, ns=None, context=None, **kwargs):
        # type: (URIRef, dict, list, AbstractFountain) -> dict
        if context is None:
            context = {}

        if ns is None:
            ns = get_ns(fountain)

        j_types = []
        data['@id'] = uri
        data['@type'] = j_types
        prefixes = dict(ns.graph.namespaces())
        for t in types:
            if isinstance(t, URIRef):
                t_n3 = t.n3(ns)
            else:
                t_n3 = t
            props = fountain.get_type(t_n3)['properties']

            short_type = t_n3.split(':')[1]
            context[short_type] = {'@id': str(extend_uri(t_n3, prefixes)), '@type': '@id'}
            j_types.append(short_type)
            for p_n3 in data:
                if p_n3 in props:
                    p = extend_uri(p_n3, prefixes)
                    pdict = fountain.get_property(p_n3)
                    if pdict['type'] == 'data':
                        range = pdict['range'][0]
                        if range == 'rdfs:Resource':
                            datatype = Literal(data[p_n3]).datatype
                        else:
                            datatype = extend_uri(range, prefixes)
                        jp = {'@type': datatype, '@id': p}
                    else:
                        jp = {'@type': '@id', '@id': p}

                    context[p_n3] = jp
                    p_n3_data = data[p_n3]
                    if isinstance(p_n3_data, dict):
                        sub = self.enrich(BNode(shortuuid.uuid()).n3(ns), p_n3_data, pdict['range'], fountain, ns,
                                          context)
                        data[p_n3] = sub['@graph']
                    elif hasattr(p_n3_data, '__call__'):
                        data[p_n3] = p_n3_data(key=p_n3, context=context, uri_provider=self.url_for, **kwargs)

        return {'@context': context, '@graph': data}
