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

from setuptools import setup, find_packages

__author__ = 'Fernando Serena'

with open("agora/metadata.json", 'r') as stream:
    metadata = json.load(stream)

setup(
    name="agora-py",
    version=metadata['version'],
    author=metadata['author'],
    author_email=metadata['email'],
    description=metadata['description'],
    license="Apache 2",
    keywords=["linked-data", "traversal", "sparql"],
    url=metadata['github'],
    download_url="https://github.com/fserena/agora-py/tarball/{}".format(metadata['version']),
    packages=find_packages(exclude=['ez_setup', 'examples', 'tests']),
    install_requires=['requests', 'flask', 'shortuuid', 'Flask-Negotiate', 'redis', 'hiredis',
                      'rdflib==4.2.1', 'networkx>=2.0', 'futures', 'rfc3987', 'python-dateutil'],
    classifiers=[],
    include_package_data=True,
    package_dir={'agora': 'agora'},
    package_data={'agora': ['metadata.json']},
)
