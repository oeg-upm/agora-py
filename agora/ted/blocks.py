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
import itertools
from abc import abstractmethod
from urlparse import urljoin

from rdflib import ConjunctiveGraph
from rdflib import Graph
from rdflib import RDF
from rdflib.term import Node, BNode

from agora.ted.evaluate import find_params
from agora.ted.ns import CORE, WOT, MAP
from agora.ted.utils import encode_rdict

__author__ = 'Fernando Serena'


def describe(graph, elm, filters=[], trace=None):
    def desc(obj=False):
        tp = (None, None, elm) if obj else (elm, None, None)
        for (s, p, o) in graph.triples(tp):
            triple = (s, p, o)
            ext_node = s if obj else o
            if triple not in trace:
                trace.add(triple)
                yield triple
            if ext_node not in trace:
                if isinstance(ext_node, BNode):
                    ignore = any(list(graph.triples((ext_node, x, None))) for x in filters)
                    if not ignore:
                        trace.add(ext_node)
                        for t in describe(graph, ext_node, filters=filters, trace=trace):
                            yield t

    if trace is None:
        trace = set([])
    for t in itertools.chain(desc()):
        yield t


class Resource(object):
    def __init__(self, graph, node):
        # type: (Graph, any) -> None
        self.__graph = ConjunctiveGraph()
        self.__types = set([])
        self.__vars = set([])
        self.__endpoints = set([])
        self.__graph = graph
        self.__triples = Graph()

        for t in describe(graph, node):
            self.__triples.add(t)

        try:
            self.__td_node = list(self.__graph.subjects(predicate=WOT.describes, object=node)).pop()
        except IndexError:
            self.__id = node.toPython()
            self.__td_node = None
        else:
            try:
                self.__id = list(self.__graph.objects(self.__td_node, WOT.identifier)).pop().toPython()
            except IndexError:
                self.__id = self.__td_node.toPython()

        self.__node = node
        self.__types = set(self.__graph.objects(self.__node, RDF.type))

        self.__access_mappings = set([])
        if self.__td_node is not None:
            for mr_node in self.__graph.objects(self.__td_node, MAP.hasAccessMapping):
                mr = AccessMapping.from_graph(self.__graph, mr_node)
                self.__access_mappings.add(mr)

            self.__vars = reduce(lambda x, y: set.union(x, y), [mr.vars for mr in self.__access_mappings], set([]))
            self.__endpoints = set([mr.endpoint for mr in self.__access_mappings])

    def endpoint_mappings(self, e):
        return reduce(lambda x, y: set.union(x, y),
                      map(lambda x: x.mappings, filter(lambda x: x.endpoint == e, self.__access_mappings)), set([]))

    @property
    def triples(self):
        return self.__triples

    @property
    def access_mappings(self):
        return frozenset(self.__access_mappings)

    @property
    def base(self):
        return frozenset(self.__endpoints)

    @property
    def types(self):
        return frozenset(self.__types)

    @property
    def graph(self):
        return self.__graph

    @property
    def node(self):
        return self.__node

    @property
    def td_node(self):
        return self.__td_node

    @property
    def id(self):
        return self.__id

    @property
    def vars(self):
        return frozenset(self.__vars)


class AccessMapping(object):
    def __init__(self):
        self.endpoint = None
        self.mappings = set([])
        self.__vars = set([])

    @staticmethod
    def from_graph(graph, node):
        # type: (Graph, Node) -> iter
        mr = AccessMapping()

        try:
            for m in graph.objects(node, MAP.hasMapping):
                mr.mappings.add(Mapping.from_graph(graph, m))
        except IndexError:
            pass

        e_node = list(graph.objects(node, MAP.mapsResourcesFrom)).pop()
        mr.endpoint = Endpoint.from_graph(graph, e_node)
        ref = mr.endpoint.href
        for param in find_params(str(ref)):
            mr.__vars.add(param)

        return mr

    @property
    def vars(self):
        return frozenset(self.__vars)


class Endpoint(object):
    def __init__(self):
        self.href = None
        self.whref = None
        self.path = None
        self.mappings = set([])
        self.media = 'application/json'

    @staticmethod
    def from_graph(graph, node):
        # type: (Graph, Node) -> iter
        endpoint = Endpoint()
        try:
            endpoint.media = list(graph.objects(node, WOT.mediaType)).pop()
        except IndexError:
            pass

        try:
            endpoint.path = list(graph.objects(node, MAP.jsonPath)).pop()
        except IndexError:
            pass

        try:
            endpoint.href = list(graph.objects(node, WOT.href)).pop()
        except IndexError:
            whref = list(graph.objects(node, WOT.withHRef)).pop()
            endpoint.whref = whref

        return endpoint

    def __add__(self, other):
        endpoint = Endpoint()
        if isinstance(other, Endpoint):
            endpoint.mappings.update(other.mappings)
            endpoint.media = other.media
            other = other.whref if other.href is None else other.href

        endpoint.href = urljoin(self.href + '/', other, allow_fragments=True)
        return endpoint


class Mapping(object):
    def __init__(self):
        self.key = None
        self.predicate = None
        self.transform = None

    @staticmethod
    def from_graph(graph, node):
        mapping = Mapping()

        try:
            mapping.predicate = list(graph.objects(node, MAP.predicate)).pop()
            mapping.key = list(graph.objects(node, MAP.key)).pop().toPython()
        except IndexError:
            pass

        try:
            mapping.transform = create_transform(graph, list(graph.objects(node, MAP.valuesTransformedBy)).pop())
        except IndexError:
            pass

        return mapping


def create_transform(graph, node):
    if list(graph.triples((node, RDF.type, WOT.ThingDescription))):
        return ResourceTransform.from_graph(graph, node)


class Transform(object):
    def __init__(self):
        pass

    def attach(self, data):
        def wrapper(*args, **kwargs):
            return self.apply(data, *args, **kwargs)

        return wrapper

    @abstractmethod
    def apply(self, data, *args, **kwargs):
        pass


class ResourceTransform(Transform):
    def __init__(self):
        super(ResourceTransform, self).__init__()
        self.resource_node = None

    @staticmethod
    def from_graph(graph, node):
        transform = ResourceTransform()
        transform.resource_node = list(graph.objects(node, WOT.describes)).pop()
        return transform

    def apply(self, data, *args, **kwargs):
        def merge(x, y):
            z = y.copy()
            z.update(x)
            return z

        if not isinstance(data, dict):
            uri_provider = kwargs['uri_provider']
            if not isinstance(data, list):
                data = [data]
            parent_item = kwargs.get('$item', None)
            base_rdict = {"$parent": parent_item} if parent_item is not None else {}
            res = [uri_provider(self.resource_node, encode_rdict(merge({"$item": v}, base_rdict))) for v in data]
            return res  # [:min(3, len(res))]
        return data


class TED(object):
    def __init__(self, graph):
        self.__ecosystem = Ecosystem(graph)

    @property
    def ecosystem(self):
        return self.__ecosystem


class Ecosystem(object):
    def __init__(self, graph):
        # type: (Graph) -> None
        try:
            self.__node = list(graph.subjects(RDF.type, CORE.Ecosystem)).pop()
        except IndexError:
            raise ValueError('Ecosystem node not found')

        self.__resources = set([])
        self.__roots = set([])

        root_nodes = set([])
        for r_node in graph.objects(self.__node, CORE.hasComponent):
            resource = Resource(graph, r_node)
            self.__resources.add(resource)
            self.__roots.add(resource)
            root_nodes.add(r_node)

        for r_node in graph.objects(predicate=WOT.describes):
            if r_node not in root_nodes:
                resource = Resource(graph, r_node)
                self.__resources.add(resource)

    @property
    def things(self):
        return frozenset(self.__resources)

    @property
    def roots(self):
        return frozenset(self.__roots)
