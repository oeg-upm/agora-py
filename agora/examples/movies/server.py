import logging

from agora import Agora, setup_logging
from agora.collector.cache import RedisCache
from agora.collector.scholar import Scholar
from agora.engine.fountain.onto import DuplicateVocabulary
from agora.examples.movies import load_films_from_dbpedia
from agora.server.fountain import build as bn
from agora.server.fragment import build as bf
from agora.server.planner import build as bp
from agora.server.sparql import build as bs

__author__ = 'Fernando Serena'

setup_logging(logging.DEBUG)

cache = RedisCache(min_cache_time=1000, persist_mode=True, base='server', path='cache',
                   redis_file='server/cache/cache.db')
agora = Agora(persist_mode=True, redis_file='server/fountain/fountain.db', base='server', path='fountain')


def query(query, **kwargs):
    return agora.query(query, collector=scholar, **kwargs)


def fragment(**kwargs):
    return agora.fragment_generator(collector=scholar, **kwargs)


server = bs(agora, query_function=query, import_name=__name__)
bf(agora, server=server, fragment_function=fragment)
bp(agora.planner, server=server)
bn(agora.fountain, server=server)

scholar = Scholar(agora.planner, cache=cache)

with open('movies.ttl') as f:
    try:
        agora.fountain.add_vocabulary(f.read())
    except DuplicateVocabulary:
        pass

# Each film URI found in dbpedia is added to Agora as a seed
for film in load_films_from_dbpedia():
    try:
        agora.fountain.add_seed(unicode(film), 'dbpedia-owl:Film')
    except Exception:
        pass

if __name__ == '__main__':
    server.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False, threaded=True)
