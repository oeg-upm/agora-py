from datetime import datetime

from agora.client import AgoraClient

__author__ = 'Fernando Serena'

agora = AgoraClient()


# Example queries
queries = ["""SELECT * WHERE {?s dbpedia-owl:starring ?actor
                              OPTIONAL { ?actor dbp:birthName ?name }
                              }"""]

# queries = ["""SELECT * WHERE {?s dbpedia-owl:starring ?actor ;
#                                  dbp:birthName ?name .
#                               }"""]

elapsed = []

for query in queries:
    pre = datetime.now()
    # Ask agora for results of the given query,
    # evaluating candidate results for each fragment triple collected (chunk_size=1)
    # -> Removing chunk_size argument forces to wait until all relevant triples are collected
    for row in agora.query(query):
        print row.asdict()
    post = datetime.now()
    elapsed.append((post - pre).total_seconds())

print elapsed
