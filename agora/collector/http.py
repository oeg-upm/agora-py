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
import math

import requests
from StringIO import StringIO
from requests.utils import parse_dict_header
import email.utils as eut
from datetime import datetime as dt

__author__ = "Fernando Serena"

log = logging.getLogger('agora.collector.http')

RDF_MIMES = {'turtle': 'text/turtle', 'xml': 'application/rdf+xml'}


def get_resource_ttl(headers):
    cache_control = headers.get('Cache-Control', None)
    ttl = None
    if cache_control is not None:
        cache_dict = parse_dict_header(cache_control)
        ttl = cache_dict.get('max-age', ttl)
        if ttl is not None:
            ttl = math.ceil(float(ttl))
    return ttl


def http_get(uri, format):
    log.debug('HTTP GET {}'.format(uri))
    try:
        response = requests.get(uri, headers={'Accept': RDF_MIMES[format]}, timeout=30)
    except requests.Timeout:
        log.debug('[Dereference][TIMEOUT][GET] {}'.format(uri))
        return True
    except UnicodeEncodeError:
        log.debug('[Dereference][ERROR][ENCODE] {}'.format(uri))
        return True
    except Exception:
        log.debug('[Dereference][ERROR][GET] {}'.format(uri))
        return True

    if response.status_code == 200:
        return StringIO(response.content), response.headers
    else:
        return response.status_code != 406


def extract_ttl(headers):
    ttl = None
    cache_control = headers.get('Cache-Control', None)
    if cache_control is not None:
        cache_dict = parse_dict_header(cache_control)
        ttl = int(math.ceil(float(cache_dict.get('max-age', ttl))))
    else:
        expires = headers.get('expires', None)
        if expires is not None:
            exp_dt = dt(*eut.parsedate(expires)[:6])
            ttl = int(math.ceil((exp_dt - dt.now()).total_seconds()))

    return ttl
