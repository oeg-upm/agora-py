import requests
from agora.collector.wrapper import ResourceWrapper
from rdflib import Graph
from rdflib import Literal
from rdflib import Namespace
from rdflib import RDF
from dateutil.parser import parse
from rdflib import URIRef
from rdflib import XSD

LIBRAIRY = Namespace('http://drinventor.dia.fi.upm.es/vocabulary#')

wrapper = ResourceWrapper('drinventor.dia.fi.upm.es', url_scheme='http')


def get_document_graph(document_uri, did):
    graph = Graph()
    document = requests.get(str(document_uri)).json()
    graph.add((document_uri, RDF.type, LIBRAIRY.Document))
    graph.add((document_uri, LIBRAIRY.identifier, Literal(did)))
    title = document.get('title', None)
    if title is not None:
        graph.add((document_uri, LIBRAIRY.title, Literal(title)))
    created = parse(document['creationTime'])
    graph.add((document_uri, LIBRAIRY.creationTime, Literal(created, datatype=XSD.datetime)))
    return graph


def get_service_graph(service_uri):
    graph = Graph()
    documents = requests.get(str(service_uri)).json()
    graph.add((service_uri, RDF.type, LIBRAIRY.DocumentService))
    for d in documents:
        d = d.replace('http://drinventor.eu/documents/', '')
        document_uri = URIRef(wrapper.url_for('get_document', did=d))
        graph.add((service_uri, LIBRAIRY.exposesDocument, document_uri))
    return graph
