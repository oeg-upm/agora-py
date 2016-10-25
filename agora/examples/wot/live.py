import logging

from agora import Agora, setup_logging
from agora.collector.cache import RedisCache
from agora.collector.scholar import Scholar
from datetime import datetime
from time import sleep

setup_logging(logging.DEBUG)

# Agora object
agora = Agora(persist_mode=True, redis_file='store/fountain/fountain.db', path='fountain')
with open('wot.ttl') as f:
    try:
        agora.fountain.add_vocabulary(f.read())
    except:
        pass

try:
    agora.fountain.add_seed('http://localhost:5005/things', 'wot:Service')
except:
    pass

q = """SELECT DISTINCT ?id (AVG(?value) as ?avg) ?lt ?lvalue WHERE {
                            [] wot:identifier ?id ;
                               wot:hasWebThingProperty [
                                   rdfs:label "tsky" ;
                                   wot:hasLatestEntry [
                                            wot:value ?lvalue ;
                                            wot:entryTimeStamp ?lt
                                        ] ;
                                   wot:hasLog [
                                        wot:hasEntry [
                                            wot:value ?value
                                        ]
                                    ]
                                ]
                          } GROUP BY ?id"""

q2 = """SELECT (COUNT(*) as ?cnt) WHERE { ?s a wot:WebThing }"""

q3 = """SELECT (COUNT(?s) as ?cnt) WHERE { ?s rdfs:label ?l . ?s wot:observedBy ?d  FILTER(STR(?l) = "mag") }"""

q4 = """SELECT ?s ?v WHERE { ?s rdfs:label "tamb" ;
                                <http://www.wot.org#observedBy> ?d ;
                                wot:hasLatestEntry [
                                    wot:value ?v ;
                                    wot:entryTimeStamp ?t
                                 ] .
                              FILTER (?v > -50)}"""

elapsed = []

for query in [q, q2, q3, q4] * 2:
    pre = datetime.now()
    for row in agora.query(query):
        for label in row.labels:
            print label + '=' + str(row[label]),
        print
    post = datetime.now()
    elapsed.append((post - pre).total_seconds())
    sleep(1)

print elapsed

raw_input()
