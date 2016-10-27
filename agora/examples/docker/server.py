import logging

from agora import Agora, setup_logging
from agora.collector.cache import RedisCache
from agora.collector.scholar import Scholar
from agora.engine.fountain.onto import DuplicateVocabulary
from agora.examples.librairy import wrapper, get_document_graph, get_service_graph
from agora.server.fountain import build as bn
from agora.server.fragment import build as bf
from agora.server.planner import build as bp
from agora.server.sparql import build as bs
from rdflib import URIRef

__author__ = 'Fernando Serena'

setup_logging(logging.INFO)

cache = RedisCache(min_cache_time=1000, persist_mode=True, base='server', path='cache',
                   redis_file='server/cache/cache.db')
agora = Agora(persist_mode=True, redis_file='server/fountain/fountain.db', base='server', path='fountain')


def query(query, **kwargs):
    return agora.query(query, collector=scholar, **kwargs)


def fragment(**kwargs):
    return agora.fragment(collector=scholar, **kwargs)


server = bs(agora, query_function=query, import_name=__name__)
bf(agora, server=server, fragment_function=fragment)
bp(agora.planner, server=server)
bn(agora.fountain, server=server)

with open('librairy.ttl') as f:
    try:
        agora.fountain.add_vocabulary(f.read())
    except DuplicateVocabulary:
        pass

librairy_cache = cache.resource_cache


@wrapper.intercept('/api/0.2/documents/<did>')
def get_document(did):
    document_uri = URIRef(wrapper.url_for(get_document, did=did))
    graph = librairy_cache.get_context(str(document_uri))
    if not graph:
        graph = get_document_graph(document_uri, did)

    return graph, {'Cache-Control': 'max-age=100'}


@wrapper.intercept('/api/0.2/documents')
def get_service():
    service_uri = URIRef(wrapper.url_for(get_service))
    graph = librairy_cache.get_context(str(service_uri))
    if not graph:
        graph = get_service_graph(service_uri)

    return graph, {'Cache-Control': 'max-age=100'}


scholar = Scholar(agora.planner, cache=cache, loader=wrapper.load)

try:
    agora.fountain.add_seed(wrapper.url_for(get_service), 'librairy:DocumentService')
except Exception:
    pass

if __name__ == '__main__':
    server.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False, threaded=True)
