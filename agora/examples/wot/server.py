import logging

from agora import Agora, setup_logging
from agora.collector.cache import RedisCache
from agora.collector.scholar import Scholar
from agora.server.fountain import build as bn
from agora.server.fragment import build as bf
from agora.server.planner import build as bp
from agora.server.sparql import build as bs

setup_logging(logging.INFO)

cache = RedisCache(min_cache_time=20, persist_mode=True, path='cache', redis_file='store/cache/cache.db')

# Agora object
agora = Agora(persist_mode=True, redis_file='store/fountain/fountain.db', path='fountain')


def query(query, **kwargs):
    # return agora.query(query)
    # return agora.query(query, cache=cache)
    return agora.query(query, collector=scholar, **kwargs)


def fragment(**kwargs):
    # return agora.fragment_generator(**kwargs)
    # return agora.fragment_generator(cache=cache, **kwargs)
    return agora.fragment_generator(collector=scholar, **kwargs)


server = bs(agora, query_function=query, import_name=__name__)
bf(agora, server=server, fragment_function=fragment)
bp(agora.planner, server=server)
bn(agora.fountain, server=server)

with open('wot.ttl') as f:
    try:
        agora.fountain.add_vocabulary(f.read())
    except:
        pass

scholar = Scholar(agora.planner, cache=cache)

try:
    agora.fountain.add_seed('http://localhost:5005/things', 'wot:Service')
except:
    pass

if __name__ == '__main__':
    server.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False, threaded=True)
