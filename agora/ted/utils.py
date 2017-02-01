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

import base64

__author__ = 'Fernando Serena'


def encode_rdict(rd, parent_item=None):
    if parent_item:
        rd["$parent"] = parent_item

    sorted_keys = sorted(rd.keys())
    sorted_fields = []
    for k in sorted_keys:
        sorted_fields.append("{}: {}".format(str(k), str(rd[k])))
    str_rd = '{' + ','.join(sorted_fields) + '}'
    return base64.b64encode(str_rd)


def encode_rdict(rd):
    sorted_keys = sorted(rd.keys())
    sorted_fields = []
    for k in sorted_keys:
        sorted_fields.append('"%s": "%s"' % (str(k), str(rd[k])))
    str_rd = '{' + ','.join(sorted_fields) + '}'
    return base64.b64encode(str_rd)
