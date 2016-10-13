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

import pbclient
from agora import Agora, setup_logging
from agora.collector.cache import RedisCache
from agora.collector.scholar import Scholar
from agora.collector.wrapper import ResourceWrapper
from agora.engine.fountain.onto import DuplicateVocabulary
from datetime import datetime
from rdflib import Graph
from rdflib import Literal
from rdflib import Namespace
from rdflib import RDF
from rdflib import URIRef
from rdflib import XSD

__author__ = 'Fernando Serena'

pbclient.set('endpoint', 'http://crowdcrafting.org')

PB = Namespace('http://pybossa.com/vocabulary#')

wrapper = ResourceWrapper('crowdcrafting.org', url_scheme='https')

# Create a cache for fragment collection
cache = RedisCache(min_cache_time=30, persist_mode=True, path='cache', redis_file='store/movies.db')

pybossa_cache = cache.resource_cache

# Agora object
agora = Agora(persist_mode=True, path='fountain', redis_file='store/fountain.db')

with open('pybossa.ttl') as f:
    try:
        agora.fountain.add_vocabulary(f.read())
    except DuplicateVocabulary:
        pass


@wrapper.intercept('/api/project/<pid>')
def get_project(pid):
    project_uri = URIRef(wrapper.url_for(get_project, pid=pid))
    graph = pybossa_cache.get_context(str(project_uri))
    if not graph:
        graph = Graph()
        project = pbclient.get_project(pid)
        graph.add((project_uri, RDF.type, PB.Project))
        graph.add((project_uri, PB.identifier, Literal(project.id)))
        graph.add((project_uri, PB.allowAnonymous, Literal(project.allow_anonymous_contributors)))
        graph.add((project_uri, PB.name, Literal(project.short_name)))
        created = datetime.strptime(project.created, '%Y-%m-%dT%H:%M:%S.%f')
        graph.add((project_uri, PB.creationDate, Literal(created, datatype=XSD.datetime)))

        last_id = 0
        n_tasks = 0
        while n_tasks <= 100:
            tasks = pbclient.get_tasks(pid, last_id=last_id)
            if not len(tasks):
                break
            n_tasks += len(tasks)
            for task in tasks:
                print 'reading task {}'.format(task.id)
                task_uri = URIRef(wrapper.url_for(get_task, pid=pid, tid=task.id))
                graph.add((project_uri, PB.offersTask, task_uri))
                last_id = task.id
    return graph, {'Cache-Control': 'max-age=100'}


@wrapper.intercept('/api/project/<pid>/tasks/<tid>')
def get_task(pid, tid):
    task_uri = URIRef(wrapper.url_for(get_task, pid=pid, tid=tid))
    graph = pybossa_cache.get_context(str(task_uri))
    if not graph:
        graph = Graph()
        task = pbclient.get_tasks(pid, limit=1, last_id=tid).pop()
        graph.add((task_uri, RDF.type, PB.Task))
        graph.add((task_uri, PB.identifier, Literal(task.id)))
        graph.add((task_uri, PB.status, Literal(task.state)))
        created = datetime.strptime(task.created, '%Y-%m-%dT%H:%M:%S.%f')
        graph.add((task_uri, PB.creationDate, Literal(created, datatype=XSD.datetime)))
    return graph, {'Cache-Control': 'max-age=100'}


def feed_projects():
    last_id = 0
    n_projects = 0
    while n_projects < 2:
        projects = pbclient.get_projects(last_id=last_id, limit=1)
        if not projects:
            break
        for p in projects:
            seed = URIRef(wrapper.url_for(get_project, pid=p.id))
            print 'adding seed {}'.format(seed)
            agora.fountain.add_seed(seed, 'pb:Project')
            last_id = max(last_id, p.id)
        n_projects += len(projects)


if not agora.fountain.get_type_seeds('pb:Project'):
    feed_projects()

# print wrapper.get('https://crowdcrafting.org/api/project/681').serialize(format='turtle')

setup_logging(logging.DEBUG)

q1 = """SELECT * WHERE { ?s a pb:Project ;
                            pb:identifier ?i ;
                            pb:creationDate ?created
                        }"""

q2 = """SELECT ?name (COUNT(?t) as ?numtask) WHERE { ?s pb:identifier ?id .
                                                     ?s pb:name ?name .
                                                     OPTIONAL { ?s pb:offersTask ?t }
                                                   } GROUP BY ?name ORDER BY ?numtask"""

q3 = """SELECT ?name ?ntask WHERE {
            SELECT ?name (COUNT(?task) as ?ntask) WHERE { ?s a pb:Project ;
                                                  pb:name ?name ;
                                                  pb:allowAnonymous ?anonym ;
                                                  pb:offersTask ?task
                                                  FILTER (?anonym)
                                             } GROUP BY ?name }
                                         HAVING (?ntask > 10 && ?ntask < 20)"""


scholar = Scholar(agora.planner, cache=cache)

elapsed = []

for query in [q2]:
    pre = datetime.now()
    for row in agora.query(query, loader=wrapper.load, collector=scholar):
        for label in row.labels:
            print label + '=' + str(row[label]),
        print
    print
    post = datetime.now()
    elapsed.append((post - pre).total_seconds())

print elapsed
