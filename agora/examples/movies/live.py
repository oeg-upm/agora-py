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

from agora import Agora, setup_logging
from agora.collector.cache import RedisCache
from agora.collector.scholar import Scholar
from agora.engine.fountain.onto import DuplicateVocabulary
from agora.examples.movies import load_films_from_dbpedia
from datetime import datetime

__author__ = 'Fernando Serena'

# Setup logging level for Agora
setup_logging(logging.DEBUG)

# Agora object
agora = Agora(persist_mode=True, redis_file='store/fountain.db')

# Open and add the vocabulary that we want to use to explore movies and associated data in dbpedia
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
