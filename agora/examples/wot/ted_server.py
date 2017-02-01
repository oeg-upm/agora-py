# coding=utf-8
import logging
from StringIO import StringIO

from rdflib import Graph

import agora.examples
from agora import Agora, setup_logging
from agora import RedisCache
from agora.collector.scholar import Scholar
from agora.server.fountain import build as bn
from agora.server.fragment import build as bf
from agora.server.planner import build as bp
from agora.server.sparql import build as bs
from agora.ted import Gateway
from agora.ted import TED
from agora.ted.publish import build as bg

setup_logging(logging.DEBUG)

with open('ted.ttl') as f:
    ted_str = f.read()

g = Graph()
g.parse(StringIO(ted_str), format='turtle')

cache = RedisCache(min_cache_time=10, persist_mode=True, path='cache', redis_file='store/cache/cache.db')

# Agora object
agora = Agora(persist_mode=True, redis_file='store/fountain/fountain.db', path='fountain')
with open('wot.ttl') as f:
    try:
        agora.fountain.add_vocabulary(f.read())
    except:
        pass

ted = TED(g)
gw = Gateway(ted, agora.fountain, server_name='localhost', server_port=5000, path='/gateway')

scholar = Scholar(agora.planner, cache=cache, loader=gw.load)


def query(query, **kwargs):
    return agora.query(query, collector=scholar, **kwargs)


def fragment(**kwargs):
    return agora.fragment_generator(collector=scholar, **kwargs)


server = bs(agora, query_function=query, import_name=__name__)
bf(agora, server=server, fragment_function=fragment)
bp(agora.planner, server=server)
bn(agora.fountain, server=server)
bg(gw, server=server)

agora.fountain.delete_type_seeds('wot:Service')
agora.fountain.delete_type_seeds('sch:Place')
for uri, type in gw.seeds:
    try:
        agora.fountain.add_seed(uri, type)
    except:
        pass

if __name__ == '__main__':
    server.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False, threaded=True)
