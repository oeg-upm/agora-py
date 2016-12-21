from datetime import datetime

from docker import Client
from rdflib import Graph
from rdflib import Literal
from rdflib import Namespace
from rdflib import RDF
from rdflib import URIRef
from rdflib import XSD

from agora.collector.wrapper import ResourceWrapper

DOCKER = Namespace('http://www.docker.com/vocabulary#')

wrapper = ResourceWrapper('localhost', url_scheme='http')

cli = Client(base_url='unix://var/run/docker.sock')


def get_docker_image(image_uri, id):
    graph = Graph()
    image = filter(lambda x: x['Id'] == id, cli.images()).pop()
    graph.add((image_uri, RDF.type, DOCKER.Image))
    graph.add((image_uri, DOCKER.identifier, Literal(id)))
    created = datetime.utcfromtimestamp(image['Created'])
    graph.add((image_uri, DOCKER.creationTime, Literal(created, datatype=XSD.datetime)))
    return graph


def get_docker_container(container_uri, id):
    graph = Graph()
    container = cli.containers(all=True, filters={'id': id}).pop()
    graph.add((container_uri, RDF.type, DOCKER.Container))
    graph.add((container_uri, DOCKER.identifier, Literal(id)))
    graph.add((container_uri, DOCKER.state, Literal(container['State'])))
    created = datetime.utcfromtimestamp(container['Created'])
    graph.add((container_uri, DOCKER.creationTime, Literal(created, datatype=XSD.datetime)))
    image_name = container['Image']
    image_id = cli.images(image_name).pop()['Id']
    image_uri = URIRef(wrapper.url_for('get_image', id=image_id))
    graph.add((container_uri, DOCKER.fromImage, image_uri))
    return graph


def get_docker_host(host_uri):
    graph = Graph()
    images = cli.images(quiet=True)
    graph.add((host_uri, RDF.type, DOCKER.Host))
    for image_id in images:
        image_uri = URIRef(wrapper.url_for('get_image', id=image_id))
        graph.add((host_uri, DOCKER.hasImage, image_uri))
    containers = cli.containers(quiet=True, all=True)
    for container in containers:
        container_uri = URIRef(wrapper.url_for('get_container', id=container['Id']))
        graph.add((host_uri, DOCKER.hasContainer, container_uri))
    return graph
