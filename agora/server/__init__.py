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
import urlparse
from functools import wraps

import requests
from flask import Flask, jsonify, request, url_for
from flask_negotiate import consumes, produces

__author__ = 'Fernando Serena'

JSON = 'application/json'
TURTLE = 'text/turtle'
HTML = 'text/html'

log = logging.getLogger('agora.server')


class APIError(Exception):
    status_code = 400

    def __init__(self, message, status_code=None, payload=None):
        Exception.__init__(self)
        self.message = str(message)
        if status_code is not None:
            self.status_code = status_code
        self.payload = payload

    def to_dict(self):
        rv = dict(self.payload or ())
        rv['message'] = self.message
        return rv


class NotFound(APIError):
    def __init__(self, message, payload=None):
        super(NotFound, self).__init__(message, 404, payload)


class Conflict(APIError):
    def __init__(self, message, payload=None):
        super(Conflict, self).__init__(message, 409, payload)


def handle_invalid_usage(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response


class AgoraServer(Flask):
    def __init__(self, import_name):
        super(AgoraServer, self).__init__(import_name)
        self.errorhandler(APIError)(handle_invalid_usage)

    def response(self, rv, content_type=JSON, status=200):
        if content_type == JSON:
            response = jsonify(rv)

        else:
            response = self.make_response(rv)
            response.headers['Content-Type'] = content_type
        response.status_code = status
        return response

    @staticmethod
    def url_for(func_name, **kwargs):
        return url_for(func_name, _external=True, **kwargs)

    def make_response(self, rv):
        return super(AgoraServer, self).make_response(rv)

    def produce(self, result, produce_types=()):
        if result is None:
            return ''
        if JSON in produce_types:
            response = jsonify(result)
        else:
            response = self.make_response(str(result))
            response.headers['Content-Type'] = produce_types[0] if produce_types else 'text/plain'
        return response

    @property
    def request_args(self):
        return request.args

    def get(self, rule, produce_types=(JSON, HTML,)):
        # type: (str, iter) -> Callable
        def decorator(f):
            @wraps(f)
            def wrap(*args, **kwargs):
                response = self.produce(f(*args, **kwargs), produce_types)
                return response

            return self.route(rule, methods=['GET'])(produces(*produce_types)(wrap))

        return decorator

    def post(self, rule, consume_types=(JSON,), produce_types=()):
        # type: (str, iter, iter) -> Callable
        def decorator(f):
            @wraps(f)
            def wrap(*args, **kwargs):
                data = request.data
                if request.content_type == JSON:
                    data = request.json
                # Request data is passed as first argument
                result = f(data, *args, **kwargs)
                response = self.produce(result, produce_types)
                response.headers['Location'] = result
                response.status_code = 201
                return response

            content_f = consumes(*consume_types)(wrap)
            if produce_types:
                content_f = produces(*produce_types)(content_f)
            return self.route(rule, methods=['POST'])(content_f)

        return decorator

    def put(self, rule, consume_types=(JSON,), produce_types=()):
        # type: (str, iter, iter) -> Callable
        def decorator(f):
            @wraps(f)
            def wrap(*args, **kwargs):
                data = request.data
                if request.content_type == JSON:
                    data = request.json
                # Request data is passed as first argument
                result = f(data, *args, **kwargs)
                response = self.produce(result, produce_types)
                return response

            content_f = consumes(*consume_types)(wrap)
            if produce_types:
                content_f = produces(*produce_types)(content_f)
            return self.route(rule, methods=['PUT'])(content_f)

        return decorator

    def delete(self, rule, produce_types=()):
        # type: (str, iter) -> Callable
        def decorator(f):
            @wraps(f)
            def wrap(*args, **kwargs):
                response = self.produce(f(*args, **kwargs), produce_types)
                return response

            return self.route(rule, methods=['DELETE'])(produces(*produce_types)(wrap))

        return decorator


class AgoraClient(object):
    def __init__(self, host='localhost', port=9002):
        self.host = 'http://{}:{}'.format(host, port)

    def _get_request(self, path, accept='application/json'):
        try:
            response = requests.get(urlparse.urljoin(self.host, path), headers={'Accept': accept})
            if response.status_code != 200:
                raise IOError(response.content)
            if accept == 'application/json':
                return response.json()
            return response.content
        except requests.ConnectionError:
            message = 'Remote service is not available'
            raise EnvironmentError(message)

    @staticmethod
    def __process_request_data(data, content_type):
        if content_type == JSON:
            return json.dumps(data)
        return str(data)

    def _post_request(self, path, data, content_type='text/turtle'):
        response = requests.post(urlparse.urljoin(self.host, path),
                                 data=self.__process_request_data(data, content_type),
                                 headers={'Content-Type': content_type})
        if response.status_code != 201:
            print response.content