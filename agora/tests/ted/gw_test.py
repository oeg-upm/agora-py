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

from agora.ted import TED, Gateway
from agora.tests.ted import TEDTest

__author__ = 'Fernando Serena'


class CreateTEDGateway(TEDTest):
    def test_create_gw(self):
        with open('agora/tests/ted/teds/ted1.ttl') as f:
            ted_str = f.read()

        g = Graph()
        g.parse(StringIO(ted_str), format='turtle')

        ted = TED(g)
        gw = Gateway(ted)
        self.log.info(gw)

        g = gw.load(gw.base + '/stars2')
        self.log.info(g.serialize(format='turtle'))
        g = gw.load(gw.base + '/stars2/tsky')
        self.log.info(g.serialize(format='turtle'))

        self.log.info(gw.seeds)
        assert isinstance(g, Graph)
