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
from contextlib import closing

import re
from agora import Agora
from agora.collector import triplify
from agora.server import Server, APIError, Client
from flask import request

__author__ = 'Fernando Serena'

log = logging.getLogger('agora.server.fragment')


def build(agora, server=None, import_name=__name__, fragment_function=None, agp_fragment_function=None):
    # type: (Agora, Server, str, Callable, Callable) -> AgoraServer

    if server is None:
        server = Server(import_name)

    fragment_function = agora.fragment if fragment_function is None else fragment_function
    agp_fragment_function = agora.agp_fragment if agp_fragment_function is None else agp_fragment_function

    @server.get('/fragment', produce_types=('text/n3', 'application/agora-quad', 'text/html'))
    def get_fragment():
        def gen_fragment():
            if request.accept_mimetypes.best == 'application/agora-quad':
                for c, s, p, o in fragment_dict['generator']:
                    quad = u'{}·{}·{}·{}\n'.format(c, s.n3(), p.n3(), o.n3())

                    yield quad
            else:
                for prefix, uri in prefixes:
                    yield '@prefix {}: <{}> .\n'.format(prefix, uri)
                yield '\n'
                for c, s, p, o in fragment_dict['generator']:
                    triple = u'{} {} {} .\n'.format(s.n3(plan.namespace_manager),
                                                    p.n3(plan.namespace_manager), o.n3(plan.namespace_manager))

                    yield triple

        try:
            query = request.args.get('query', None)
            if query is not None:
                fragment_dict = fragment_function(query)
            else:
                agp_str = request.args.get('gp')
                agp_match = re.search(r'\{(.*)\}', agp_str).groups(0)
                if len(agp_match) != 1:
                    raise APIError('Invalid graph pattern')

                agp = re.split('\. ', agp_match[0])
                fragment_dict = agp_fragment_function(*[tp.strip() for tp in agp])
            plan = fragment_dict['plan']
            prefixes = fragment_dict['prefixes']
            return gen_fragment()
        except Exception, e:
            raise APIError(e.message)

    return server


class FragmentClient(Client):
    def __init__(self, host='localhost', port=5000):
        super(FragmentClient, self).__init__(host, port)

    def fragment(self, query):
        return self._get_request('fragment?query={}'.format(query), accept='text/n3')

    def agp_fragment(self, *tps):
        quads_gen = self._get_request('fragment?gp={ %s }' % ' . '.join(tps), accept='application/agora-quad')
        with closing(quads_gen) as gen:
            for quad in gen:
                quad = str(tuple(quad.split('\xc2\xb7')))
                yield triplify(quad)


def client(host='localhost', port=5000):
    # type: (str, int) -> FragmentClient
    return FragmentClient(host, port)
