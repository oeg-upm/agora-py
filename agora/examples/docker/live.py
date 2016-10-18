import logging

from agora import Agora, setup_logging
from agora.engine.fountain.onto import DuplicateVocabulary
from agora.examples.docker import get_docker_host, wrapper, get_docker_image, get_docker_container
from datetime import datetime
from rdflib import URIRef

# Agora object
agora = Agora(persist_mode=True, redis_file='store/fountain/fountain.db', path='fountain')

setup_logging(logging.DEBUG)

with open('docker.ttl') as f:
    try:
        agora.fountain.add_vocabulary(f.read())
    except DuplicateVocabulary:
        pass


@wrapper.intercept('/images/<id>')
def get_image(id):
    image_uri = URIRef(wrapper.url_for(get_image, id=id))
    graph = get_docker_image(image_uri, id)
    return graph, {}


@wrapper.intercept('/containers/<id>')
def get_container(id):
    container_uri = URIRef(wrapper.url_for(get_container, id=id))
    graph = get_docker_container(container_uri, id)
    return graph, {}


@wrapper.intercept('/')
def get_host():
    host_uri = URIRef(wrapper.url_for(get_host))
    graph = get_docker_host(host_uri)
    return graph, {}


try:
    agora.fountain.add_seed(wrapper.url_for(get_host), 'docker:Host')
except Exception:
    pass

q1 = """SELECT * WHERE { [] a docker:Host ;
                            docker:hasImage [
                                docker:identifier ?i ;
                                docker:creationTime ?ts ;
                            ]
                        }"""

q2 = """SELECT * WHERE { [] a docker:Container ;
                            docker:identifier ?container ;
                            docker:fromImage [
                                docker:identifier ?image
                            ]
                        }"""

q3 = """SELECT * WHERE { [] a docker:Container;
                            docker:identifier ?container ;
                            docker:state "running"
                        }"""

elapsed = []

for query in [q3] * 5:
    pre = datetime.now()
    for row in agora.query(query, loader=wrapper.load):
        print row.asdict()
    print
    post = datetime.now()
    elapsed.append((post - pre).total_seconds())

print elapsed
