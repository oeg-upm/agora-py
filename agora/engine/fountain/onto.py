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

import StringIO
import logging
import urlparse

from rdflib import Graph, RDF, URIRef
from rdflib.namespace import OWL
from rdflib.plugins.parsers.notation3 import BadSyntax

from agora.engine.fountain.schema import Schema

__author__ = 'Fernando Serena'

log = logging.getLogger('agora.engine.fountain.onto')


class VocabularyError(Exception):
    pass


class DuplicateVocabulary(VocabularyError):
    pass


class VocabularyNotFound(VocabularyError):
    pass


class UnknownVocabulary(VocabularyError):
    pass


def __load_owl(owl):
    """

    :param owl: The ontology to be loaded to the fountain
    :return:
    """
    raw_g = Graph()
    for f in ['turtle', 'xml']:
        try:
            raw_g.parse(source=StringIO.StringIO(owl), format=f)
            log.debug('Parsed ontology in {} format'.format(f))
            break
        except SyntaxError, e:
            log.warn(e.message)
            pass

    if not len(raw_g):
        raise VocabularyError()

    owl_g = Graph()
    owl_g.bind('owl', OWL)
    for s, p, o in raw_g.triples((None, None, None)):
        if (isinstance(s, URIRef) and s.startswith('file')) or p.startswith('file') or (
                    isinstance(o, URIRef) and o.startswith('file')):
            continue
        owl_g.add((s, p, o))

    found_ontos = list(owl_g.subjects(RDF.type, OWL.Ontology))
    if len(found_ontos) != 1:
        raise VocabularyNotFound("Incorrect number of ontology statements: {}".format(len(found_ontos)))

    uri = found_ontos.pop()

    for p, u in raw_g.namespaces():
        if not p.startswith('ns'):
            owl_g.bind(p, u)

    vid = [p for (p, u) in owl_g.namespaces() if uri in u and p != '']
    imports = owl_g.objects(uri, OWL.imports)
    if not len(vid):
        vid = urlparse.urlparse(uri).path.split('/')[-1]
    else:
        vid = vid.pop()

    # (identifier, ontology uri, graph, imports)
    return vid, uri, owl_g, imports


def add_vocabulary(schema, owl):
    # type: (Schema, str) -> iter
    vid, uri, owl_g, imports = __load_owl(owl)

    if vid in schema.contexts:
        raise DuplicateVocabulary('Vocabulary already contained')

    schema.add_context(vid, owl_g)
    vids = [vid]

    # Add imported vocabularies
    for im_uri in imports:
        log.debug('Importing {} from {}...'.format(im_uri, vid))
        im_g = Graph()
        try:
            im_g.load(im_uri, format='turtle')
        except BadSyntax:
            try:
                im_g.load(im_uri)
            except BadSyntax:
                log.error('Bad syntax in {}'.format(im_uri))
        except Exception:
            log.warn("Couldn't import {}".format(im_uri))
        else:
            try:
                child_vids = add_vocabulary(schema, im_g.serialize(format='turtle'))
                vids.extend(child_vids)
            except DuplicateVocabulary, e:
                log.debug('vocabulary already added: {}'.format(im_uri))
            except VocabularyNotFound, e:
                log.warning('uri not found for {}'.format(im_uri))
            except Exception, e:
                log.error(e.message)

    return vids


def update_vocabulary(schema, vid, owl):
    """

    :param vid:
    :param owl:
    :return:
    """
    owl_vid, uri, owl_g, imports = __load_owl(owl)

    if vid != owl_vid:
        raise Exception("Identifiers don't match")

    if vid not in schema.contexts:
        raise UnknownVocabulary('Vocabulary id is not known')

    schema.update_context(vid, owl_g)


def delete_vocabulary(schema, vid):
    if vid not in schema.contexts:
        raise UnknownVocabulary('Vocabulary id is not known')

    schema.remove_context(vid)


def get_vocabularies(schema):
    """

    :return:
    """
    return schema.contexts


def get_vocabulary(schema, vid):
    """

    :param vid:
    :return:
    """
    return schema.get_context(vid).serialize(format='turtle')
