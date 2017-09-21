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
import traceback
from Queue import Queue, Empty
from datetime import datetime
from threading import Thread

from flask import redirect
from flask import render_template
from flask import request
from rdflib import BNode
from rdflib import URIRef

from agora import Agora
from agora.engine.utils import Semaphore
from agora.server import Server, APIError, Client

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
        if value.datatype is not None:
            return 'typed-literal'
        return 'literal'


def result(row):
    def r_dict(l):
        value = row[l]
        type = value_type(value)
        value_p = value.toPython()
        if isinstance(value_p, datetime):
            value_p = str(value_p)
        res = {"type": type, "value": value_p}
        if 'literal' in type:
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
        def gen_thread(status):
            first = True
            try:
                for row in gen:
                    if first:
                        queue.put('{\n')
                        queue.put('  "head": %s,\n  "results": {\n    "bindings": [\n' % json.dumps(head(row)))
                        first = False
                    else:
                        queue.put(',\n')
                    queue.put('      {}'.format(json.dumps(result(row)).encode('utf-8')))
                if first:
                    queue.put('{\n')
                    queue.put('  "head": [],\n  "results": {\n    "bindings": []\n  }\n')
                else:
                    queue.put('\n    ]\n  }\n')
                queue.put('}')
            except Exception, e:
                exception = e

            status['completed'] = True

        def gen_queue(status):
            with stop_event:
                while not status['completed'] or not queue.empty():
                    status['last'] = datetime.now()
                    try:
                        for chunk in queue.get(timeout=1.0):
                            yield chunk
                    except Empty:
                        if not status['completed']:
                            yield '\n'
                        elif status['exception']:
                            raise APIError(status['exception'].message)

        try:
            query = request.args.get('query')
            incremental = json.loads(request.args.get('incremental', 'true'))
            kwargs = dict(request.args.items())
            del kwargs['query']
            if 'incremental' in kwargs:
                del kwargs['incremental']

            stop_event = Semaphore()
            gen = query_function(query, incremental=incremental, stop_event=stop_event, **kwargs)
            queue = Queue()
            request_status = {
                'completed': False,
                'exception': None
            }
            stream_th = Thread(target=gen_thread, args=(request_status,))
            stream_th.daemon = False
            stream_th.start()

            return gen_queue(request_status)
        except Exception, e:
            traceback.print_exc()
            raise APIError(e.message)

    @server.route('/')
    def index():
        return render_template('base.html')

    @server.route('/<tmpl>')
    def map(tmpl):
        try:
            return render_template(tmpl)
        except Exception:
            return redirect('/')

    return server


class SPARQLClient(Client):
    def __init__(self, host='localhost', port=5000):
        super(SPARQLClient, self).__init__(host, port)

    def query(self, query):
        return self._get_request('sparql?query={}'.format(query), accept='application/sparql-results+json')


def client(host='localhost', port=5000):
    # type: (str, int) -> FountainClient
    return SPARQLClient(host, port)
