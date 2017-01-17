import logging
from StringIO import StringIO
from datetime import datetime
from time import sleep

from rdflib import Graph

from agora import Agora, setup_logging
import agora.examples
from agora import RedisCache
from agora.collector.scholar import Scholar
from agora.ted import TED, Gateway

setup_logging(logging.INFO)

with open('../../tests/ted/teds/ted1.ttl') as f:
    ted_str = f.read()

g = Graph()
g.parse(StringIO(ted_str), format='turtle')

cache = RedisCache(min_cache_time=10, persist_mode=True, path='cache', redis_file='store/cache/cache.db')

# Agora object
agora = Agora(persist_mode=True, redis_file='store/fountain/fountain.db', path='fountain')
with open('wot2.ttl') as f:
    try:
        agora.fountain.add_vocabulary(f.read())
    except:
        pass

ted = TED(g)
gw = Gateway(ted, agora.fountain)


agora.fountain.delete_type_seeds('wot:Thing')
for uri, type in gw.seeds:
    try:
        agora.fountain.add_seed(uri, type)
    except:
        pass

q10 = """SELECT * WHERE { [] wot:identifier ?name ;
                            wot:offersInteraction [
                               rdfs:label ?l ;
                               wot:hasLatestEntry [ wot:value ?v ]
                            ] }"""

q11 = """SELECT * WHERE { [] wot:identifier ?name ;
                            wot:location [
                               foaf:name ?city ;
                               geo:lat ?lat ;
                               geo:long ?long
                            ] ;
                            wot:offersInteraction [
                               rdfs:label ?l ;
                               wot:hasLatestEntry [ wot:value ?v ]
                            ]
                          FILTER (STR(?l)="tsky")}"""

q12 = """SELECT DISTINCT * WHERE {
                            ?w wot:identifier ?name .
                            ?w wot:offersInteraction ?i .
                            ?i rdfs:label ?l .
                            ?i wot:hasLatestEntry ?e .
                            ?e wot:value ?v .
                            ?e wot:entryTimeStamp ?t .
                        }"""

q13 = """SELECT DISTINCT * WHERE {
                            ?uri a sch:Place ;
                               foaf:name ?place .
                            }"""


# scholar = Scholar(agora.planner, cache=cache, loader=gw.load)

elapsed = []

for query in [q12]:
    pre = datetime.now()
    n = 0
    for row in agora.query(query, loader=gw.load):
        print '[', (datetime.now() - pre).total_seconds(), '] solution:',
        for label in row.labels:
            print label + '=' + str(row[label]),
        print
        n += 1
    print n, 'solutions'
    post = datetime.now()
    elapsed.append((post - pre).total_seconds())
    # sleep(0.5)

print elapsed

raw_input()
