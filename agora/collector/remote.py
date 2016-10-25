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
from agora.collector import AbstractCollector
from agora.server.fragment import client as fc
from agora.server.planner import client as pc

__author__ = "Fernando Serena"


class RemoteCollector(AbstractCollector):
    def __init__(self, host='localhost', port=9002, planner=None):
        # type: (str, int, AbstractPlanner) -> RemoteCollector
        self.__planner = pc(host, port) if planner is None else planner
        self.__fragment = fc(host, port)

    def get_fragment_generator(self, agp, **kwargs):
        # type: (AGP, dict) -> dict
        plan = self.__planner.make_plan(agp)
        generator = self.__fragment.agp_fragment(agp)
        fragment_dict = {'prefixes': self.prefixes, 'plan': plan, 'generator': generator}
        return fragment_dict

    @property
    def prefixes(self):
        return self.__planner.fountain.prefixes
