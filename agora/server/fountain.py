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

from agora.engine.fountain import AbstractFountain
from agora.engine.fountain.onto import VocabularyNotFound, DuplicateVocabulary, VocabularyError
from agora.engine.fountain.seed import InvalidSeedError, DuplicateSeedError
from agora.server import Server, APIError, Conflict, TURTLE, NotFound, Client

__author__ = 'Fernando Serena'

log = logging.getLogger('agora.server.fountain')


def build(fountain, server=None, import_name=__name__):
    # type: (AbstractFountain, Server, str) -> AgoraServer

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

    @server.get('/types/<string:type>')
    def get_type(type):
        try:
            return fountain.get_type(type)
        except TypeError as e:
            raise NotFound(e.message)

    @server.get('/vocabs/<string:vid>', produce_types=(TURTLE,))
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
        try:
            return fountain.get_paths(elm)
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
        response = self._get_request('prefixes')
        return response

    def update_vocabulary(self, vid, owl):
        raise NotImplementedError

    def get_paths(self, elm):
        response = self._get_request('paths/{}'.format(elm))
        return response

    @property
    def seeds(self):
        response = self._get_request('seeds')
        return response

    def delete_type_seeds(self, type):
        raise NotImplementedError

    def get_property(self, property):
        if property not in self.__properties:
            self.__properties[property] = self._get_request('properties/{}'.format(property))

        return self.__properties[property]

    def get_vocabulary(self, vid):
        raise NotImplementedError

    def delete_vocabulary(self, vid):
        raise NotImplementedError

    def get_type(self, type):
        if type not in self.__types:
            self.__types[type] = self._get_request('types/{}'.format(type))
        return self.__types[type]

    @property
    def types(self):
        response = self._get_request('types')
        return response.get('types')

    @property
    def vocabularies(self):
        raise NotImplementedError

    def add_seed(self, uri, type):
        response = self._post_request('seeds',
                                      {'uri': uri,
                                       'type': type},
                                      content_type='application/json')
        return response

    def delete_seed(self, sid):
        raise NotImplementedError

    def get_seed_type_digest(self, type):
        response = self._get_request('seeds/{}/digest'.format(type))
        return response

    def add_vocabulary(self, owl):
        response = self._post_request('vocabs', owl)
        return response


def client(host='localhost', port=5000):
    # type: (str, int) -> FountainClient
    return FountainClient(host, port)
