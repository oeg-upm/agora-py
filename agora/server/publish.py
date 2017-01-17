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
import traceback
from contextlib import closing

import re
from time import sleep

from rdflib import Graph
from rdflib import URIRef

from agora import Agora
from agora import RedisCache
from agora.collector import triplify
from agora.engine.plan import AGP
from agora.engine.plan.agp import TP
from agora.engine.plan.graph import AGORA
from agora.server import Server, APIError, Client
from flask import request, jsonify, url_for

__author__ = 'Fernando Serena'

log = logging.getLogger('agora.server.publish')


def build(cache, server=None, import_name=__name__, fragment_function=None):
    # type: (RedisCache, Server, str, Callable, Callable) -> AgoraServer

    if server is None:
        server = Server(import_name)

    @server.get('/resources', produce_types=('text/turtle', 'text/html'))
    def get_resources():
        store = cache.resource_cache
        uri = request.args.get('uri', None)
        if uri is None:
            g = Graph()
            container_uri = URIRef(url_for('get_resources', _external=True))
            for c in store.contexts():
                if isinstance(c.identifier, URIRef):
                    r_uri = container_uri + '?uri=' + c.identifier
                    g.add((container_uri, AGORA.hasResource, r_uri))
        else:
            g = store.get_context(uri)
            if not g:
                return 'not found'

        return g.serialize(format='turtle')

    return server
