import logging
from datetime import datetime

from agora.client import AgoraClient
from agora import setup_logging

__author__ = 'Fernando Serena'

setup_logging(logging.DEBUG)

agora = AgoraClient()

q1 = """SELECT * WHERE { ?s librairy:title "Model reduction for real-time fluids" ;
                            librairy:creationTime ?created
                        }"""


print agora.fragment(q1).serialize(format='turtle')

elapsed = []

for query in [q1]:
    pre = datetime.now()
    for row in agora.query(query):
        print row.asdict()
    print
    post = datetime.now()
    elapsed.append((post - pre).total_seconds())

print elapsed
