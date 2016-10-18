import logging

from agora import Agora, setup_logging
from agora.engine.fountain.onto import DuplicateVocabulary
from agora.examples.librairy import wrapper, get_document_graph, get_service_graph
from datetime import datetime
from rdflib import URIRef

# Agora object
agora = Agora(persist_mode=True, redis_file='store/fountain/fountain.db', path='fountain')

setup_logging(logging.DEBUG)

with open('librairy.ttl') as f:
    try:
        agora.fountain.add_vocabulary(f.read())
    except DuplicateVocabulary:
        pass


@wrapper.intercept('/api/0.2/documents/<did>')
def get_document(did):
    document_uri = URIRef(wrapper.url_for(get_document, did=did))
    graph = get_document_graph(document_uri, did)
    return graph, {}


@wrapper.intercept('/api/0.2/documents')
def get_service():
    service_uri = URIRef(wrapper.url_for(get_service))
    graph = get_service_graph(service_uri)
    return graph, {}


try:
    agora.fountain.add_seed(wrapper.url_for(get_service), 'librairy:DocumentService')
except Exception:
    pass

q1 = """SELECT (COUNT(DISTINCT ?title) as ?cnt) WHERE { [] a librairy:Document ;
                            librairy:title ?title ;
                            librairy:creationTime ?created }"""

elapsed = []

for query in [q1]:
    pre = datetime.now()
    for row in agora.query(query, loader=wrapper.load):
        print row.asdict()
    print
    post = datetime.now()
    elapsed.append((post - pre).total_seconds())

print elapsed
