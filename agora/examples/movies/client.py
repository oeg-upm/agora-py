from agora.client import AgoraClient

__author__ = 'Fernando Serena'

client = AgoraClient()

for row in client.query(
        """SELECT * WHERE { ?s a dbpedia-owl:Film } LIMIT 10"""):
    print row
