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
from threading import Lock

from flask import request
from rdflib import URIRef
from shortuuid import uuid

from agora.engine.fountain import AbstractFountain
from agora.engine.fountain.onto import VocabularyNotFound, DuplicateVocabulary, VocabularyError
from agora.engine.fountain.seed import InvalidSeedError, DuplicateSeedError
from agora.server import Server, APIError, Conflict, TURTLE, NotFound, Client, HTML, tuples_force_seed, JSON, \
    dict_force_seed

__author__ = 'Fernando Serena'

log = logging.getLogger('agora.server.fountain')


def build(fountain, server=None, import_name=__name__):
    # type: (AbstractFountain, Server, str) -> Server

    if server is None:
        server = Server(import_name)

    @server.get('/seeds/id/<string:sid>')
    def get_seed(sid):
        try:
            return fountain.get_seed(sid)
        except InvalidSeedError, e:
            raise APIError(e.message)

    @server.delete('/seeds/<string:type>')
    def delete_type_seeds(type):
        try:
            fountain.delete_type_seeds(type)
        except TypeError as e:
            raise NotFound(e.message)

    @server.get('/prefixes')
    def prefixes():
        return fountain.prefixes

    @server.post('/prefixes')
    def add_prefixes(prefixes):
        fountain.add_prefixes(prefixes)

    @server.get('/types/<string:type>')
    def get_type(type):
        try:
            return fountain.get_type(type)
        except TypeError as e:
            raise NotFound(e.message)

    @server.get('/vocabs/<string:vid>', produce_types=(TURTLE, HTML))
    def get_vocabulary(vid):
        return fountain.get_vocabulary(vid)

    @server.get('/types')
    def types():
        return {'types': fountain.types}

    @server.delete('/seeds/id/<string:sid>')
    def delete_seed(sid):
        try:
            fountain.delete_seed(sid)
        except InvalidSeedError, e:
            raise NotFound(e.message)

    @server.post('/seeds')
    def add_seed(seed_dict):
        try:
            sid = fountain.add_seed(seed_dict['uri'], seed_dict['type'])
            return server.url_for('get_seed', sid=sid)
        except (TypeError, ValueError) as e:
            raise APIError(e.message)
        except DuplicateSeedError as e:
            raise Conflict(e.message)

    @server.get('/paths/<string:elm>')
    def get_paths(elm):
        force_seed = request.args.getlist('force_seed', None)
        force_seed = None if not force_seed else map(lambda x: ('http://{}'.format(uuid()), x), force_seed)
        try:
            return fountain.get_paths(elm, force_seed=force_seed)
        except TypeError, e:
            raise APIError(e.message)

    @server.post('/paths', produce_types=(JSON,), consume_types=(JSON,))
    def get_paths_force_seeds(req_json):
        force_seed = list(tuples_force_seed(req_json['force_seed']))
        try:
            return fountain.get_paths(req_json['elm'], force_seed=force_seed)
        except TypeError, e:
            raise APIError(e.message)

    @server.get('/seeds/<string:type>')
    def get_type_seeds(type):
        try:
            return fountain.get_type_seeds(type)
        except TypeError as e:
            raise NotFound(e.message)

    @server.get('/seeds/<string:type>/digest')
    def get_seed_type_digest(type):
        try:
            return {'digest': fountain.get_seed_type_digest(type)}
        except TypeError as e:
            raise NotFound(e.message)

    @server.delete('/vocabs/<string:vid>')
    def delete_vocabulary(vid):
        fountain.delete_vocabulary(vid)

    @server.get('/vocabs')
    def vocabularies():
        return fountain.vocabularies

    @server.get('/properties/<string:property>')
    def get_property(property):
        try:
            return fountain.get_property(property)
        except TypeError as e:
            raise NotFound(e.message)

    @server.post('/vocabs', consume_types=('text/turtle',))
    def add_vocabulary(owl):
        try:
            return fountain.add_vocabulary(owl)
        except VocabularyNotFound, e:
            raise APIError('Ontology URI not found: {}'.format(e.message))
        except DuplicateVocabulary, e:
            raise Conflict(e.message)
        except VocabularyError, e:
            raise APIError(e.message)

    @server.get('/seeds')
    def seeds():
        return fountain.seeds

    @server.get('/properties')
    def properties():
        return {'properties': fountain.properties}

    @server.put('/vocabs/<string:vid>', consume_types=('text/turtle',))
    def update_vocabulary(owl, vid):
        try:
            fountain.update_vocabulary(vid, owl)
        except VocabularyNotFound, e:
            raise APIError('Ontology URI not found: {}'.format(e.message))
        except VocabularyError, e:
            raise APIError(e.message)

    return server


class FountainClient(Client, AbstractFountain):
    def __init__(self, host='localhost', port=5000):
        super(FountainClient, self).__init__(host, port)
        self.__types = {}
        self.__properties = {}
        self.__prefixes = {}
        self.__lock = Lock()

    @property
    def properties(self):
        response = self._get_request('properties')
        return response.get('properties')

    def get_type_seeds(self, type):
        response = self._get_request('seeds/{}'.format(type))
        return response.get('seeds')

    def get_seed(self, sid):
        response = self._get_request('seeds/id/{}'.format(sid))
        return response

    @property
    def prefixes(self):
        prefixes_raw = self._get_request('prefixes')
        return {prefix: URIRef(prefixes_raw[prefix]) for prefix in prefixes_raw}

    def add_prefixes(self, prefixes):
        self._post_request('/prefixes', data=prefixes, content_type='application/json')

    def update_vocabulary(self, vid, owl):
        response = self._put_request('vocabs/{}'.format(vid), owl)
        return response

    def get_paths(self, elm, force_seed=None):
        if force_seed:
            url = 'paths'
            type_seed_dict = dict_force_seed(force_seed)
            req_json = {'elm': str(elm), 'force_seed': type_seed_dict}
            response = self._post_request(url, req_json, content_type='application/json', accept='application/json')
        else:
            url = 'paths/%s' % elm
            response = self._get_request(url, accept='application/json')

        return response

    @property
    def seeds(self):
        response = self._get_request('seeds')
        return response

    def delete_type_seeds(self, type):
        response = self._delete_request('seeds/{}'.format(type))
        return response

    def get_property(self, property):
        try:
            return self._get_request('properties/{}'.format(property))
        except IOError as e:
            raise TypeError(e.message['text'])

    def get_vocabulary(self, vid):
        return self._get_request('vocabs/{}'.format(vid), accept='text/turtle')

    def delete_vocabulary(self, vid):
        return self._delete_request('vocabs/{}'.format(vid))

    def get_type(self, type):
        try:
            return self._get_request('types/{}'.format(type))
        except IOError as e:
            raise TypeError(e.message['text'])

    @property
    def types(self):
        response = self._get_request('types')
        return response.get('types')

    @property
    def vocabularies(self):
        response = self._get_request('vocabs')
        return response

    def add_seed(self, uri, type):
        response = self._post_request('seeds',
                                      {'uri': uri,
                                       'type': type},
                                      content_type='application/json')
        return response

    def connected(self, source, target):
        url = 'paths/{}'.format(target)
        url += '?force_seed={}'.format(source)
        response = self._get_request(url)
        paths = response.get('paths', [])
        if len(paths) == 1:
            path = paths.pop()
            connected = path['cycles'] or path['steps']
        else:
            connected = len(paths) > 1
        return bool(connected)

    def delete_seed(self, sid):
        raise NotImplementedError

    def get_seed_type_digest(self, type):
        response = self._get_request('seeds/{}/digest'.format(type))
        return response

    def add_vocabulary(self, owl):
        try:
            response = self._post_request('vocabs', owl)
            return response
        except IOError as e:
            if e.message['code'] == 409:
                raise DuplicateVocabulary(e.message['text'])
            raise e


def client(host='localhost', port=5000):
    # type: (str, int) -> FountainClient
    return FountainClient(host, port)
