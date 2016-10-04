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

from agora.collector.execution import PlanExecutor

__author__ = "Fernando Serena"


class Collector(object):
    def __init__(self, planner, cache=None):
        # type: (AbstractPlanner, Cache) -> Collector
        self.cache = cache
        self.__planner = planner

    def get_fragment(self, *tps, **kwargs):
        """
        Return a complete fragment for a given gp.
        :param gp: A graph pattern
        :return:
        """
        plan = self.__planner.make_plan(*tps)
        executor = PlanExecutor(plan)
        return executor.get_fragment(**kwargs)

    def get_fragment_generator(self, *tps, **kwargs):
        """
        Return a fragment generator for a given gp.
        :param gp:
        :param kwargs:
        :return:
        """
        plan = self.__planner.make_plan(*tps)
        executor = PlanExecutor(plan)
        gen, prefixes, plan = executor.get_fragment_generator(cache=self.cache, **kwargs)
        return {'generator': gen, 'prefixes': prefixes, 'plan': plan}

    @property
    def prefixes(self):
        return self.__planner.fountain.prefixes
