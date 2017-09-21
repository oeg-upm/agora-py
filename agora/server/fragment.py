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
import re
import traceback
from Queue import Queue, Empty
from contextlib import closing
from datetime import datetime
from threading import Thread

from flask import request

from agora import Agora
from agora.collector import triplify
from agora.engine.plan import AGP
from agora.engine.plan.agp import TP
from agora.engine.utils import Semaphore
from agora.server import Server, APIError, Client

__author__ = 'Fernando Serena'

log = logging.getLogger('agora.server.fragment')


def build(agora, server=None, import_name=__name__, fragment_function=None):
    # type: (Agora, Server, str, callable) -> Server

    if server is None:
        server = Server(import_name)

    fragment_function = agora.fragment_generator if fragment_function is None else fragment_function

    @server.get('/fragment',
                produce_types=('text/n3', 'application/agora-quad', 'application/agora-quad-min', 'text/html'))
    def get_fragment():
        def gen_thread(status):
            try:
                first = True
                min_quads = '-min' in best_mime
                if best_mime.startswith('application/agora-quad'):
                    for c, s, p, o in generator:
                        if min_quads:
                            quad = u'{}·{}·{}·{}\n'.format(c, s.n3(plan.namespace_manager),
                                                           p.n3(plan.namespace_manager), o.n3(plan.namespace_manager))
                        else:
                            quad = u'{}·{}·{}·{}\n'.format(c, s.n3(), p.n3(), o.n3())

                        queue.put(quad)
                else:
                    if first:
                        for prefix, uri in prefixes.items():
                            queue.put('@prefix {}: <{}> .\n'.format(prefix, uri))
                        queue.put('\n')
                    for c, s, p, o in generator:
                        triple = u'{} {} {} .\n'.format(s.n3(plan.namespace_manager),
                                                        p.n3(plan.namespace_manager), o.n3(plan.namespace_manager))

                        queue.put(triple)
            except Exception as e:
                status['exception'] = e

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
            stop_event = Semaphore()
            query = request.args.get('query', None)
            kwargs = dict(request.args.items())
            if query is not None:

                del kwargs['query']
                fragment_dict = fragment_function(query=query, stop_event=stop_event, **kwargs)
            else:
                tps_str = request.args.get('agp')
                tps_match = re.search(r'\{(.*)\}', tps_str).groups(0)
                if len(tps_match) != 1:
                    raise APIError('Invalid graph pattern')

                tps = re.split('\. ', tps_match[0])
                agp = AGP([tp.strip() for tp in tps], prefixes=agora.planner.fountain.prefixes)
                fragment_dict = fragment_function(agps=[agp], **kwargs)
            plan = fragment_dict['plan']
            generator = fragment_dict['generator']
            prefixes = fragment_dict['prefixes']
            best_mime = request.accept_mimetypes.best
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

    return server


class FragmentClient(Client):
    def __init__(self, host='localhost', port=5000):
        super(FragmentClient, self).__init__(host, port)

    def fragment(self, query):
        # type: (str) -> iter
        quads_gen = self._get_request('fragment?query={}'.format(query), accept='application/agora-quad')
        with closing(quads_gen) as gen:
            for quad in gen:
                quad = str(tuple(quad.split('\xc2\xb7')))
                tp_str, s, p, o = triplify(quad)
                yield TP.from_string(tp_str), s, p, o

    def agp_fragment(self, agp):
        # type: (AGP) -> iter
        quads_gen = self._get_request('fragment?agp=%s' % agp, accept='application/agora-quad')
        with closing(quads_gen) as gen:
            for quad in gen:
                quad = str(tuple(quad.split('\xc2\xb7')))
                tp_str, s, p, o = triplify(quad)
                yield TP.from_string(tp_str), s, p, o


def client(host='localhost', port=5000):
    # type: (str, int) -> FragmentClient
    return FragmentClient(host, port)
