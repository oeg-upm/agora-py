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

from rdflib import URIRef, BNode, Graph
from rdflib.namespace import RDFS, OWL, NamespaceManager

from agora.engine.utils.cache import Cache, ContextGraph, cached

__author__ = 'Fernando Serena'

log = logging.getLogger('agora.engine.fountain.path')


def __flat_slice(iterable):
    # type: (iter) -> set
    lst = filter(lambda x: x, list(iterable))
    for i, _ in enumerate(lst):
        while hasattr(lst[i], "__iter__") and not isinstance(lst[i], basestring):
            lst[i:i + 1] = lst[i]
    return set(filter(lambda x: x is not None, lst))


def __q_name(ns, term):
    # type: (NamespaceManager, any) -> any
    n3_method = getattr(term, "n3", None)
    if callable(n3_method):
        return term.n3(ns)
    return term


def __query(graph, q):
    # type: (Graph, str) -> set
    graph.namespace_manager.bind('rdfs', RDFS, replace=True, override=True)
    graph.namespace_manager.bind('owl', OWL, replace=True, override=True)
    result = graph.query(q)
    return set([__q_name(graph.namespace_manager, x) for x in __flat_slice(result)])


def __extend_prefixed(ns, pu):
    # type: (dict, str) -> URIRef

    parts = pu.split(':')
    if len(parts) == 1:
        parts = ('', parts[0])
    try:
        return URIRef(ns[parts[0]] + parts[1])
    except KeyError:
        return BNode(pu)


def __extend_with(f, graph, *args):
    # type: (callable, Graph, iter) -> iter
    args = __flat_slice(args)
    extension = __flat_slice([f(graph, t) for t in args])
    return set.union(args, extension)


def _contexts(graph):
    # type: (ContextGraph) -> list
    return [str(x.identifier) for x in graph.contexts()]


def _update_context(graph, vid, g):
    # type: (ContextGraph, str, Graph) -> None
    context = graph.get_context(vid)
    if isinstance(context, SchemaGraph):
        context = context.g
    graph.remove_context(context)
    _add_context(graph, vid, g)


def _remove_context(graph, vid):
    # type: (ContextGraph, str) -> None
    context = graph.get_context(vid)
    if isinstance(context, SchemaGraph):
        context = context.g
    graph.remove_context(context)


def _get_context(graph, vid):
    # type: (ContextGraph, str) -> Graph
    return graph.get_context(vid)


def _add_context(graph, vid, g):
    # type: (ContextGraph, str, Graph) -> None
    def match_ns(term):
        filter_ns = [ns for ns in rev_ns if ns in term]
        if filter_ns:
            ns = filter_ns.pop()
            if rev_ns[ns]:
                vid_context.bind(rev_ns[ns], ns)
            del rev_ns[ns]

    rev_ns = {ns: prefix for prefix, ns in g.namespaces() if not prefix.startswith('ns')}
    vid_context = graph.get_context(vid)
    for s, p, o in g.triples((None, None, None)):
        if o != OWL.Ontology:
            match_ns(s)
            match_ns(p)
            match_ns(o)
        vid_context.add((s, p, o))


def _prefixes(graph):
    # type: (Graph) -> dict
    return dict(graph.namespaces())


def _get_types(graph):
    # type: (Graph) -> set
    return __query(graph,
                   """SELECT DISTINCT ?c WHERE {
                        {
                            ?p a owl:ObjectProperty .
                            {
                                { ?p rdfs:range ?c }
                                UNION
                                { ?p rdfs:domain ?c }
                            }
                        }
                        UNION
                        {
                            ?p a owl:DatatypeProperty .
                            ?p rdfs:domain ?c .
                        }
                        UNION
                        { ?c a owl:Class }
                        UNION
                        { ?c a rdfs:Class }
                        UNION
                        { [] rdfs:subClassOf ?c }
                        UNION
                        { ?c rdfs:subClassOf [] }
                        UNION
                        {
                            ?r a owl:Restriction ;
                               owl:onProperty ?p .
                            {
                                ?p a owl:ObjectProperty .
                                { ?r owl:allValuesFrom ?c }
                                UNION
                                { ?r owl:someValuesFrom ?c }
                            }
                            UNION
                            { ?r owl:onClass ?c }
                        }
                        FILTER(isURI(?c))
                      }""")


def _get_properties(graph):
    # type: (Graph) -> set
    return __query(graph, """SELECT DISTINCT ?p WHERE {
                                { ?p a rdf:Property }
                                UNION
                                { ?p a owl:ObjectProperty }
                                UNION
                                { ?p a owl:DatatypeProperty }
                                UNION
                                {
                                    [] a owl:Restriction ;
                                       owl:onProperty ?p .
                                }
                                FILTER(isURI(?p))
                              }""")


def _is_object_property(graph, prop):
    # type: (Graph, str) -> bool
    evidence = __query(graph, """ASK {
                                    { %s a owl:ObjectProperty }
                                    UNION
                                    {
                                        ?r owl:onProperty %s .
                                        {
                                            { ?c a owl:Class }
                                            UNION
                                            { ?c rdfs:subClassOf ?r }
                                        }
                                        {
                                            {
                                               ?r owl:onClass ?c .
                                            }
                                            UNION
                                            {
                                                ?r owl:someValuesFrom ?c .
                                            }
                                            UNION
                                            {
                                                ?r owl:allValuesFrom ?c .
                                            }
                                        }
                                    }
                                   }""" % (prop, prop))

    return False if not evidence else bool(evidence.pop())


def _get_property_domain(graph, prop):
    # type: (Graph, str) -> set
    all_property_domains = graph.query("""
                          PREFIX owl: <http://www.w3.org/2002/07/owl#>
                          PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                          SELECT DISTINCT ?p ?c WHERE {
                             { ?p rdfs:domain ?c }
                             UNION
                             { ?c rdfs:subClassOf [ owl:onProperty ?p ] }
                             FILTER (isURI(?p) && isURI(?c))
                           }""")

    dom = map(lambda x: __q_name(graph.namespace_manager, x.c),
              filter(lambda x: __q_name(graph.namespace_manager, x.p) == prop, all_property_domains))
    return __extend_with(_get_subtypes, graph, dom)


def _get_property_range(graph, prop):
    # type: (Graph, str) -> set
    all_property_ranges = graph.query("""
                                        PREFIX owl: <http://www.w3.org/2002/07/owl#>
                                        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                                        SELECT DISTINCT ?p ?r WHERE {
                                              {?p rdfs:range ?r}
                                              UNION
                                              {
                                                    ?d owl:onProperty ?p.
                                                    { ?d owl:allValuesFrom ?r }
                                                    UNION
                                                    { ?d owl:someValuesFrom ?r }
                                                    UNION
                                                    { ?d owl:onClass ?r }
                                                    UNION
                                                    { ?d owl:onDataRange ?r }
                                              }
                                              FILTER(isURI(?p) && isURI(?r))
                                            }""")

    rang = map(lambda x: __q_name(graph.namespace_manager, x.r),
               filter(lambda x: __q_name(graph.namespace_manager, x.p) == prop, all_property_ranges))
    return __extend_with(_get_subtypes, graph, rang)


def _get_property_inverses(graph, prop):
    # type: (Graph, str) -> set
    return __query(graph, """SELECT DISTINCT ?i WHERE {
                                 {%s owl:inverseOf ?i}
                                 UNION
                                 {?i owl:inverseOf %s}
                                 FILTER(isURI(?i))
                               }""" % (prop, prop))


def _get_property_constraints(graph, prop):
    # type: (Graph, str) -> set
    all_property_domains = graph.query("""
                               PREFIX owl: <http://www.w3.org/2002/07/owl#>
                               PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                               SELECT DISTINCT ?p ?c WHERE {
                                 { ?p rdfs:domain ?c }
                                 UNION
                                 { ?c rdfs:subClassOf [ owl:onProperty ?p ] }
                                 FILTER (isURI(?p) && isURI(?c))
                               }""")

    dom = map(lambda x: __q_name(graph.namespace_manager, x.c),
              filter(lambda x: __q_name(graph.namespace_manager, x.p) == prop, all_property_domains))
    dom_supertypes = [_get_supertypes(graph, d) for d in dom]
    for d, s in zip(dom, dom_supertypes):
        if set.intersection(set(s), set(dom)):
            cons_range = __query(graph, """SELECT DISTINCT ?r WHERE {
                                              {
                                                    %s rdfs:subClassOf ?d .
                                                    ?d owl:onProperty %s .
                                                    { ?d owl:allValuesFrom ?r }
                                                    UNION
                                                    { ?d owl:someValuesFrom ?r }
                                                    UNION
                                                    { ?d owl:onClass ?r }
                                                    UNION
                                                    { ?d owl:onDataRange ?r }
                                              }
                                              FILTER(isURI(?r))
                                            }""" % (d, prop))
            cons_range = __extend_with(_get_subtypes, graph, cons_range)
            if cons_range:
                try:
                    is_strong_constraint = __query(graph, """ASK {                                                          
                                                                %s rdfs:subClassOf ?d .
                                                                ?d owl:onProperty %s .
                                                                ?d owl:allValuesFrom []                                                               
                                                            }""" % (d, prop)).pop()
                except KeyError:
                    is_strong_constraint = True

                if is_strong_constraint:
                    yield (d, list(cons_range))


def _get_supertypes(graph, ty):
    # type: (Graph, str) -> set
    res = map(lambda x: __q_name(graph.namespace_manager, x), filter(lambda y: isinstance(y, URIRef),
                                                                     graph.transitive_objects(
                                                                         __extend_prefixed(_prefixes(graph), ty),
                                                                         RDFS.subClassOf)))
    return set(filter(lambda x: str(x) != ty, res))


def _get_subtypes(graph, ty):
    # type: (Graph, str) -> set
    res = map(lambda x: __q_name(graph.namespace_manager, x), filter(lambda y: isinstance(y, URIRef),
                                                                     graph.transitive_subjects(RDFS.subClassOf,
                                                                                               __extend_prefixed(
                                                                                                   _prefixes(graph),
                                                                                                   ty))))

    return set(filter(lambda x: str(x) != ty, res))


def _get_type_properties(graph, ty):
    # type: (Graph, str) -> set
    all_class_props = graph.query("""
                                    PREFIX owl: <http://www.w3.org/2002/07/owl#>
                                    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                                    SELECT DISTINCT ?c ?p WHERE {
                                        {?c rdfs:subClassOf [ owl:onProperty ?p ] }
                                        UNION
                                        {?p rdfs:domain ?c}
                                        FILTER (isURI(?p) && isURI(?c))
                                      }""")

    all_types = __extend_with(_get_supertypes, graph, ty)
    return set([__q_name(graph.namespace_manager, r.p) for r in all_class_props if
                __q_name(graph.namespace_manager, r.c) in all_types])


def _get_type_specific_properties(graph, ty):
    # type: (Graph, str) -> set
    all_class_props = graph.query("""
                                        PREFIX owl: <http://www.w3.org/2002/07/owl#>
                                        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                                        SELECT DISTINCT ?c ?p WHERE {
                                            {?c rdfs:subClassOf [ owl:onProperty ?p ] }
                                            UNION
                                            {?p rdfs:domain ?c}
                                            FILTER (isURI(?p) && isURI(?c))
                                          }""")

    return set([__q_name(graph.namespace_manager, r.p) for r in all_class_props if
                __q_name(graph.namespace_manager, r.c) == ty])


def _get_type_references(graph, ty):
    # type: (Graph, str) -> set
    all_class_props = graph.query("""
                                    PREFIX owl: <http://www.w3.org/2002/07/owl#>
                                    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                                    SELECT ?c ?p WHERE {
                                        { ?r owl:onProperty ?p .
                                          {?r owl:someValuesFrom ?c}
                                          UNION
                                          {?r owl:allValuesFrom ?c}
                                          UNION
                                          {?r owl:onClass ?c}
                                        }
                                        UNION
                                        {?p rdfs:range ?c}
                                        FILTER (isURI(?p) && isURI(?c))
                                       }""")

    all_types = __extend_with(_get_supertypes, graph, ty)
    return set([__q_name(graph.namespace_manager, r.p) for r in all_class_props if
                __q_name(graph.namespace_manager, r.c) in all_types])


def _get_type_specific_references(graph, ty):
    # type: (Graph, str) -> set
    all_class_props = graph.query("""
                                    PREFIX owl: <http://www.w3.org/2002/07/owl#>
                                    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                                    SELECT ?c ?p WHERE {
                                        { ?r owl:onProperty ?p .
                                          {?r owl:someValuesFrom ?c}
                                          UNION
                                          {?r owl:allValuesFrom ?c}
                                          UNION
                                          {?r owl:onClass ?c}
                                        }
                                        UNION
                                        {?p rdfs:range ?c}
                                        FILTER (isURI(?p) && isURI(?c))
                                       }""")

    return set([__q_name(graph.namespace_manager, r.p) for r in all_class_props if
                __q_name(graph.namespace_manager, r.c) == ty])


def _context(f):
    # type: (callable) -> callable
    def wrap(self=None, *args, **kwargs):
        return cached(self.cache)(f)(self, *args, **kwargs)

    return wrap


@_context
def _query(graph, q):
    return graph.query(q)


class SchemaGraph(object):
    def __init__(self, query_f, g):
        self.g = g
        self.__query = query_f

    def __getattr__(self, item):
        try:
            return self.__getattribute__(item)
        except:
            return self.g.__getattribute__(item)

    def query(self, q):
        return self.__query(self.g, q)

    def get_context(self, c):
        return SchemaGraph(self.__query, self.g.get_context(c))


class Schema(object):
    def __init__(self):
        self.__cache = Cache()
        self.__graph = None
        self.__namespaces = {}
        self.__prefixes = {}

    @property
    def cache(self):
        # type: () -> Cache
        return self.__cache

    @property
    def graph(self):
        # type: () -> ContextGraph
        return SchemaGraph(self.__query, self.__graph)

    @graph.setter
    def graph(self, g):
        self.__graph = g
        self.__graph.store.graph_aware = False
        self.update_ns_dicts()

    def update_ns_dicts(self):
        self.__namespaces.update([(uri, prefix) for (prefix, uri) in self.__graph.namespaces()])
        self.__prefixes.update([(prefix, uri) for (prefix, uri) in self.__graph.namespaces()])
        self.__cache.clear()

    def add_context(self, id, context):
        # type: (str, Graph) -> None
        _add_context(self.graph, id, context)
        self.update_ns_dicts()

    def update_context(self, id, context):
        # type: (str, Graph) -> None
        _update_context(self.graph, id, context)
        self.update_ns_dicts()

    def remove_context(self, id):
        # type: (str) -> None
        _remove_context(self.graph, id)
        self.update_ns_dicts()
        self.__cache.clear()

    @property
    def contexts(self):
        # type: () -> iter
        return _contexts(self.graph)

    def get_context(self, id):
        # type: (str) -> Graph
        return _get_context(self.graph, id)

    @property
    def prefixes(self):
        # type: () -> dict
        return self.__prefixes

    @_context
    def get_types(self, context=None):
        # type: (object) -> iter
        if not isinstance(context, ContextGraph):
            context = self.graph.get_context(context) if context is not None else self.graph
        return _get_types(context)

    @_context
    def get_properties(self, context=None):
        # type: (object) -> iter
        if not isinstance(context, ContextGraph):
            context = self.graph.get_context(context) if context is not None else self.graph
        return _get_properties(context)

    @_context
    def is_object_property(self, p):
        # type: (str, Graph) -> bool
        return _is_object_property(self.graph, p)

    @_context
    def get_property_domain(self, p):
        # type: (str, Graph) -> iter
        return _get_property_domain(self.graph, p)

    @_context
    def get_property_range(self, p):
        # type: (str, Graph) -> iter
        return _get_property_range(self.graph, p)

    @_context
    def get_property_inverses(self, p):
        # type: (str, Graph) -> iter
        return _get_property_inverses(self.graph, p)

    @_context
    def get_property_constraints(self, p):
        # type: (str, Graph) -> iter
        return _get_property_constraints(self.graph, p)

    @_context
    def get_supertypes(self, t):
        # type: (str, Graph) -> iter
        return _get_supertypes(self.graph, t)

    @_context
    def get_subtypes(self, t):
        # type: (str, Graph) -> iter
        return _get_subtypes(self.graph, t)

    @_context
    def get_type_properties(self, t):
        # type: (str, Graph) -> iter
        return _get_type_properties(self.graph, t)

    @_context
    def get_type_specific_properties(self, t):
        # type: (str, Graph) -> iter
        return _get_type_specific_properties(self.graph, t)

    @_context
    def get_type_references(self, t):
        # type: (str, Graph) -> iter
        return _get_type_references(self.graph, t)

    @_context
    def get_type_specific_references(self, t):
        # type: (str, Graph) -> iter
        return _get_type_specific_references(self.graph, t)

    @_context
    def __query(self, g, q):
        return g.query(q)
