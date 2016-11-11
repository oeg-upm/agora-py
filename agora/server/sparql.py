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
import json
import logging
from datetime import datetime

from agora import Agora
from agora.server import Server, APIError, Client
from flask import request
from rdflib import BNode
from rdflib import URIRef

__author__ = 'Fernando Serena'

log = logging.getLogger('agora.server.sparql')


def head(row):
    return {'vars': list(row.labels)}


def value_type(value):
    if isinstance(value, URIRef):
        return 'uri'
    elif isinstance(value, BNode):
        return 'bnode'
    else:
        return 'literal'


def result(row):
    def r_dict(l):
        value = row[l]
        type = value_type(value)
        value_p = value.toPython()
        if isinstance(value_p, datetime):
            value_p = str(value_p)
        res = {"type": type, "value": value_p}
        if type == 'literal':
            if value.datatype:
                res['datatype'] = value.datatype.toPython()
            if value.language:
                res['xml:lang'] = str(value.language)
        return res

    return {l: r_dict(l) for l in row.labels if row[l] is not None}


def build(agora, server=None, import_name=__name__, query_function=None):
    # type: (Agora, Server, str) -> AgoraServer

    if server is None:
        server = Server(import_name)

    query_function = agora.query if query_function is None else query_function

    @server.get('/sparql', produce_types=('application/sparql-results+json', 'text/html'))
    def query():
        def gen_results():
            yield '{\n'
            first = True
            for row in gen:
                if first:
                    yield '  "head": %s,\n  "results": {\n    "bindings": [\n' % json.dumps(head(row))
                    first = False
                else:
                    yield ',\n'
                yield '      {}'.format(json.dumps(result(row)).encode('utf-8'))
            if not first:
                yield '\n    ]\n  }\n'
            yield '}'

        try:
            query = request.args.get('query')
            chunk_size = request.args.get('chunk', None)
            gen = query_function(query, chunk_size=chunk_size)
            return gen_results()
        except Exception, e:
            raise APIError(e.message)

    return server


class SPARQLClient(Client):
    def __init__(self, host='localhost', port=5000):
        super(SPARQLClient, self).__init__(host, port)

    def query(self, query):
        return self._get_request('sparql?query={}'.format(query), accept='application/sparql-results+json')


def client(host='localhost', port=5000):
    # type: (str, int) -> FountainClient
    return SPARQLClient(host, port)
