from SPARQLWrapper import SPARQLWrapper, JSON

sparql = SPARQLWrapper("http://localhost:5000/sparql")
sparql.setQuery("""SELECT * WHERE { ?s rdfs:label "tamb" ;
                                        wot:hasLatestEntry [
                                            wot:value ?v
                                         ]
                                    FILTER (?v > 0 && ?v < 40)
                                   }""")
sparql.setReturnFormat(JSON)

results = sparql.query().convert()

for result in results["results"]["bindings"]:
    print(result["s"]["value"]),
    print(result["v"]["value"])
