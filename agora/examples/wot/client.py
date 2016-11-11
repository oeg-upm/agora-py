from datetime import datetime
import agora.examples
from agora.client import AgoraClient

__author__ = 'Fernando Serena'

agora = AgoraClient()

# Example queries
queries = ["""SELECT * WHERE {?s wot:value ?v ; wot:entryTimeStamp ?t }"""]

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
                              FILTER (?v > 20)}"""

q5 = """SELECT (COUNT(?s) as ?cnt) WHERE { ?s rdfs:label "mag" . ?s wot:observedBy ?d }"""


elapsed = []

# fragment = agora.fragment(queries[0])
# print fragment.serialize(format='turtle')
# print len(fragment)

for query in [q]:
    pre = datetime.now()
    # Ask agora for results of the given query,
    # evaluating candidate results for each fragment triple collected (chunk_size=1)
    # -> Removing chunk_size argument forces to wait until all relevant triples are collected
    for row in agora.query(query):
        print row.asdict()
    post = datetime.now()
    elapsed.append((post - pre).total_seconds())

print elapsed
