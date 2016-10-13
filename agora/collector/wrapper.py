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
from urlparse import urlparse, urljoin

from werkzeug.routing import Rule, Map


class ResourceWrapper(object):
    def __init__(self, server_name, url_scheme, server_port=None):
        if server_port is None:
            server_port = '80' if url_scheme == 'http' else '443'
        else:
            server_port = str(server_port)

        self.__environ = {'SERVER_NAME': server_name,
                          'SERVER_PORT': server_port,
                          'wsgi.url_scheme': url_scheme,
                          'REQUEST_METHOD': 'GET'}

        self.__url_map = Map()
        self.__adapter = self.__url_map.bind_to_environ(self.__environ)
        self.__base = '{}://{}'.format(url_scheme, server_name)
        if server_port is not None and server_port != '80' and server_port != '443':
            self.__base += ':{}'.format(server_port)

    def add_rule(self, rule, callback):
        r = Rule(rule, endpoint=callback)
        self.__url_map.add(r)

    def url_for(self, callback, **values):
        return urljoin(self.__base, self.__adapter.build(callback, values))

    @property
    def base(self):
        return self.__base

    def __match(self, uri):
        parse = urlparse(uri, allow_fragments=True)
        if parse.hostname == self.__environ['SERVER_NAME'] and parse.scheme == self.__environ['wsgi.url_scheme'] and (
                        parse.port == self.__environ['SERVER_PORT'] or (parse.port is None)):
            return self.__adapter.match(parse.path)
        raise ValueError

    def intercept(self, rule):
        def decorator(f):
            self.add_rule(rule, f)
            return f

        return decorator

    def join(self, path):
        return urljoin(self.__base, path, allow_fragments=True)

    def load(self, uri, format=None):
        try:
            f, kwargs = self.__match(uri)
            return f(**kwargs)
        except ValueError:
            return

    def get(self, uri):
        # type: (str) -> Graph
        return self.load(uri)[0]
