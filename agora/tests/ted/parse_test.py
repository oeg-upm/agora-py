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
from StringIO import StringIO

from rdflib import Graph

from agora.ted import TED
from agora.tests.ted import TEDTest

__author__ = 'Fernando Serena'


class ParseTED(TEDTest):
    def test_parse_ttl(self):
        with open('agora/tests/ted/teds/ted1.ttl') as f:
            ted_str = f.read()

        g = Graph()
        g.parse(StringIO(ted_str), format='turtle')
        ted = TED(g)

        assert len(list(ted.ecosystem.things)) == 2
        for thing in ted.ecosystem.things:
            for i in thing.interactions:
                for e in i.endpoints:
                    self.log.info(e.uri)

