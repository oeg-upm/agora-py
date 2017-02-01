# coding=utf-8
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

from flask import request, url_for
from rdflib import Graph
from rdflib import URIRef

from agora.engine.plan.graph import AGORA
from agora.server import Server
from agora.ted.gateway import Gateway

__author__ = 'Fernando Serena'

log = logging.getLogger('agora.ted.publish')


def build(gateway, server=None, import_name=__name__):
    # type: (Gateway, Server, str) -> AgoraServer

    if server is None:
        server = Server(import_name)

    def serialize(g):
        turtle = g.serialize(format='turtle')
        gw_host = gateway.host + '/'
        if gw_host != request.host_url:
            turtle = turtle.replace(gateway.host + '/', request.host_url)
        return turtle

    @server.get(gateway.path, produce_types=('text/turtle', 'text/html'))
    def get_gateway():
        g = Graph()
        gateway_uri = URIRef(url_for('get_gateway', _external=True))
        for s_uri, type in gateway.seeds:
            r_uri = URIRef(s_uri)
            g.add((gateway_uri, AGORA.hasSeed, r_uri))

        return serialize(g)

    @server.get('{}/<path:rid>'.format(gateway.path), produce_types=('text/turtle', 'text/html'))
    def get_gw_resource(rid):
        g, headers = gateway.load(gateway.base + '/' + rid)
        return serialize(g)

    return server
