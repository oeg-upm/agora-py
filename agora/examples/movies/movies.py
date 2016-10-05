"""
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#
  Ontology Engineering Group
        http://www.oeg-upm.net/
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#
  Copyright (C) 2016 Ontology Engineering Group.
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#
  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

            http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#
"""

import logging
from datetime import datetime
from SPARQLWrapper import SPARQLWrapper, JSON
from agora import Agora, setup_logging
from agora.collector.cache import RedisCache

__author__ = 'Fernando Serena'

# Setup logging level for Agora
setup_logging(logging.INFO)


def load_films_from_dbpedia():
    """
    Get movie resources from dbpedia
    :return: movies generator
    """
    sparql = SPARQLWrapper("http://es.dbpedia.org/sparql")
    sparql.setReturnFormat(JSON)

    sparql.setQuery("""
           SELECT distinct ?film
           WHERE {?film a dbpedia-owl:Film} LIMIT 100
       """)
    results = sparql.query().convert()

    for result in results["results"]["bindings"]:
        yield result["film"]["value"]


# Create a cache for fragment collection
cache = RedisCache(min_cache_time=30)

# Agora object
agora = Agora()

# Open and add the vocabulary that we want to use to explore movies and associated data in dbpedia
with open('movies.ttl') as f:
    agora.fountain.add_vocabulary(f.read())

# Each film URI found in dbpedia is added to Agora as a seed
for film in load_films_from_dbpedia():
    try:
        agora.fountain.add_seed(unicode(film), 'dbpedia-owl:Film')
    except ValueError:
        pass

# Example queries
queries = ["""SELECT * WHERE {?s dbpedia-owl:starring [
                                    dbp:birthName ?actor
                                 ]
                              }""",
           """SELECT * WHERE {?s dbp:birthName ?name}"""]

elapsed = []

for query in queries:
    pre = datetime.now()
    # Ask agora for results of the given query,
    # evaluating candidate results for each fragment triple collected (chunk_size=1)
    # -> Removing chunk_size argument forces to wait until all relevant triples are collected
    for row in agora.query(query, cache=cache, chunk_size=1):
        for label in row.labels:
            print label + '=' + row[label],
        print
    post = datetime.now()
    elapsed.append((post - pre).total_seconds())

print elapsed
