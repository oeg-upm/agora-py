import logging
from datetime import datetime
from time import sleep

from agora import Agora, setup_logging
import agora.examples

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

q2 = """SELECT ?s WHERE { ?s a wot:WebThing }"""

q3 = """SELECT (COUNT(?s) as ?cnt) WHERE { ?s rdfs:label ?l . ?s wot:observedBy ?d  FILTER(STR(?l) = "mag") }"""

q4 = """SELECT ?s ?v ?d WHERE { ?s rdfs:label "tamb" ;
                                <http://www.wot.org#observedBy> ?d ;
                                wot:hasLatestEntry [
                                    wot:value ?v
                                 ] .
                              FILTER (?v>17.0)}"""

q5 = """SELECT ?s ?d WHERE { ?s rdfs:label "mag" . ?s wot:observedBy ?d }"""

q6 = """SELECT * WHERE { ?s a wot:WebThing ; wot:identifier ?i ; wot:encapsulatesSystem ?sys }"""

q7 = """SELECT * WHERE {
                            ?w wot:identifier "stars1" .
                            ?w wot:hasWebThingProperty ?wp .
                            ?wp rdfs:label ?prop .
                            ?wp wot:hasLatestEntry ?le .
                            ?le wot:value ?lvalue .
                            ?le wot:entryTimeStamp ?lt .
                            ?wp wot:hasLog ?log .
                            ?log wot:hasEntry ?loge .
                            ?loge wot:value ?value .
                          FILTER (STR(?prop)="tamb")}"""

q8 = """SELECT ?s ?v ?d WHERE { ?s rdfs:label ?prop ;
                                <http://www.wot.org#observedBy> ?d ;
                                wot:hasLatestEntry [
                                    wot:value ?v
                                 ] .
                              FILTER (STR(?prop)="tamb")}"""


q9 = """SELECT * WHERE {
                            ?w wot:hasWebThingProperty ?wp .
                            ?wp rdfs:label "tsky" .
                            ?wp wot:hasLatestEntry ?le .
                            ?le wot:value ?lvalue FILTER(?lvalue > 0) .
                            ?le wot:entryTimeStamp ?lt .
                            ?wp wot:hasLog ?log .
                            ?log wot:hasEntry ?loge .
                            ?loge wot:value ?value }"""


q10 = """SELECT * WHERE { ?s rdfs:label ?l }"""

elapsed = []


fgen = agora.fragment_generator(q10)['generator']
for tp, s, p, o in fgen:
    print s.n3(), p.n3(), o.n3()
#
# for query in [q10]:
#     pre = datetime.now()
#     n = 0
#     for row in agora.query(query, incremental=True):
#         print '[', (datetime.now() - pre).total_seconds(), '] solution:',
#         for label in row.labels:
#             print label + '=' + str(row[label]),
#         print
#         n += 1
#     print n, 'solutions'
#     post = datetime.now()
#     elapsed.append((post - pre).total_seconds())
#
# print elapsed

# raw_input()
