import logging

from SPARQLWrapper import JSON
from SPARQLWrapper import SPARQLWrapper
from datetime import datetime

import requests
from agora import Agora, setup_logging
from agora.collector.scholar import Scholar
from agora.collector.cache import RedisCache
from agora.collector.wrapper import ResourceWrapper
from agora.engine.fountain.onto import DuplicateVocabulary
from dateutil.parser import parse
from rdflib import Graph
from rdflib import Literal
from rdflib import Namespace
from rdflib import RDF
from rdflib import URIRef
from rdflib import XSD
from agora.server.sparql import build as bs
from agora.server.fragment import build as bf
from agora.server.planner import build as bp
from agora.server.fountain import build as bn

__author__ = 'Fernando Serena'

setup_logging(logging.INFO)

cache = RedisCache(min_cache_time=1000, persist_mode=True, base='server', path='cache',
                   redis_file='server/cache/cache.db')
agora = Agora(persist_mode=True, redis_file='server/fountain/fountain.db', base='server', path='fountain')


def query(query, **kwargs):
    return agora.query(query, collector=scholar, **kwargs)


def fragment(query, **kwargs):
    return agora.fragment(query, collector=scholar, **kwargs)


def agp_fragment(*tps, **kwargs):
    return agora.agp_fragment(collector=scholar, *tps, **kwargs)


server = bs(agora, query_function=query, import_name=__name__)
bf(agora, server=server, fragment_function=fragment, agp_fragment_function=agp_fragment)
bp(agora.planner, server=server)
bn(agora.fountain, server=server)

scholar = Scholar(agora.planner, cache=cache)

with open('movies.ttl') as f:
    try:
        agora.fountain.add_vocabulary(f.read())
    except DuplicateVocabulary:
        pass


def load_films_from_dbpedia():
    """
    Get movie resources from dbpedia
    :return: movies generator
    """
    sparql = SPARQLWrapper("http://es.dbpedia.org/sparql")
    sparql.setReturnFormat(JSON)

    sparql.setQuery("""
           SELECT distinct ?film
           WHERE {?film a dbpedia-owl:Film} LIMIT 10
       """)
    results = sparql.query().convert()

    for result in results["results"]["bindings"]:
        yield result["film"]["value"]

# Each film URI found in dbpedia is added to Agora as a seed
for film in load_films_from_dbpedia():
    try:
        agora.fountain.add_seed(unicode(film), 'dbpedia-owl:Film')
    except Exception:
        pass

if __name__ == '__main__':
    server.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False, threaded=True)
