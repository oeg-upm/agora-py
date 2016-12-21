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
from rdflib import Namespace
from rdflib import RDF
from rdflib.term import Node, BNode

__author__ = 'Fernando Serena'

WOT = Namespace('http://www.wot.org#')


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
        # type: (Graph, Node) -> None
        self.__node = node
        self.__graph = ConjunctiveGraph()
        self.__types = set([])

        # for t in describe(graph, node, filters=[WOT.onEndpoint]):
        for t in describe(graph, node):
            self.__graph.add(t)

        self.__types = set(self.__graph.objects(self.__node, RDF.type))

        self._endpoints = set([])
        for e_node in self.__graph.objects(self.__node, WOT.onEndpoint):
            for endpoint in Endpoint.from_graph(self.__graph, e_node):
                self._endpoints.add(endpoint)

    @property
    def base(self):
        return frozenset(self._endpoints)

    @property
    def types(self):
        return frozenset(self.__types)

    @property
    def graph(self):
        return self.__graph

    @property
    def node(self):
        return self.__node


class Endpoint(object):
    def __init__(self):
        self.uri = None
        self.href = None
        self.path = None
        self.mappings = set([])
        self.media = 'application/json'

    @staticmethod
    def from_graph(graph, node):
        # type: (Graph, Node, frozenset) -> iter
        endpoint = Endpoint()
        try:
            endpoint.media = list(graph.objects(node, WOT.mediaType)).pop()
        except IndexError:
            pass

        try:
            endpoint.path = list(graph.objects(node, WOT.jsonPath)).pop()
        except IndexError:
            pass

        try:
            for m in graph.objects(node, WOT.mapping):
                endpoint.mappings.add(Mapping.from_graph(graph, m))
        except IndexError:
            pass

        try:
            endpoint.uri = list(graph.objects(node, WOT.uri)).pop()
            yield endpoint
        except IndexError:
            for href in graph.objects(node, WOT.withHRef):
                endpoint.href = href
                yield endpoint

    def __add__(self, other):
        endpoint = Endpoint()
        if isinstance(other, Endpoint):
            endpoint.mappings.update(other.mappings)
            endpoint.media = other.media
            other = other.href if other.uri is None else other.uri

        endpoint.uri = urljoin(self.uri + '/', other, allow_fragments=True)
        return endpoint


class Mapping(object):
    def __init__(self):
        self.key = None
        self.uri = None
        self.transform = None

    @staticmethod
    def from_graph(graph, node):
        mapping = Mapping()

        try:
            mapping.key = list(graph.objects(node, WOT.key)).pop().toPython()
            mapping.uri = list(graph.objects(node, WOT.uri)).pop()
        except IndexError:
            pass

        try:
            mapping.transform = create_transform(graph, list(graph.objects(node, WOT.valueTransform)).pop())
        except IndexError:
            pass

        return mapping


def create_transform(graph, node):
    if list(graph.triples((node, WOT.onEndpoint, None))):
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
        transform.resource_node = node
        return transform

    def apply(self, data, *args, **kwargs):
        if not isinstance(data, dict):
            uri_provider = kwargs['uri_provider']
            resource_uri = uri_provider(self.resource_node.toPython())
            if not isinstance(data, list):
                data = [data]
            return ['{}?$item={}'.format(resource_uri, v) for v in data]
        return data

