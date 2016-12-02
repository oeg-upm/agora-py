/**
 * Created by fserena on 28/11/16.
 */

function createNode(rdf, elm) {
    if (isLiteral(elm)) {
        let value;
        let language = null;
        let datatype = null;
        if (elm.indexOf("^^") > 0) {
            let parts = elm.split('^^');
            value = parts[0].substring(1, parts[0].length - 1);
            datatype = parts[1].substring(1, parts[1].length - 1)
        } else {
            let parts = elm.split('@');
            value = parts[0];
            if (value[0] === '"') {
                value = value.substring(1, value.length - 1)
            }
            language = parts[1];
        }
        return rdf.createLiteral(value, language, datatype);
    } else if (isURI(elm)) {
        return rdf.createNamedNode(elm.substring(1, elm.length - 1));
    } else {
        let fakeURI = 'http://agora.org/fake/' + elm;
        return rdf.createNamedNode(fakeURI);
    }
}

function isLiteral(elm) {
    return elm[0] != '<' && elm[0] != '_'
}
function isURI(elm) {
    return elm[0] == '<'
}


(function () {

    'use strict';

    angular.module('AgoraApp', ['ngRoute', 'ngAnimate'])
        .config(['$routeProvider', '$locationProvider',
            function ($routeProvider, $locationProvider) {
                $routeProvider
                    .when('/', {
                        templateUrl: 'sparql.html',
                        controller: 'SPARQLController',
                        controllerAs: 'sparql'
                    });

                $locationProvider.html5Mode(true);
            }])
        .controller('SPARQLController', ['$scope', '$log', '$http', '$timeout', '$q',
            function ($scope, $log, $http, $timeout, $q) {
                $scope.solutions = false;
                $scope.fragmentSolutions = false;
                $scope.query = 'PREFIX wot: <http://www.wot.org#> \
SELECT * WHERE {?s rdfs:label ?l ; wot:hasLatestEntry [ wot:value ?v ] }';
                console.log('hello from SPARQL!');
                $scope.triples = undefined;

                // set options and call the d3sparql.xxxxx visualization methods in this library ...
                let config = {
                    "selector": "#result"
                };

                $scope.request = function () {
                    $scope.triples = undefined;
                    d3sparql.query('http://localhost:5000/sparql', $scope.query, function (json) {
                        console.log(json);
                        d3sparql.htmltable(json, config);
                        $timeout(function () {
                            $scope.solutions = true;
                        }, 0);
                    });
                };

                $scope.fragment = function () {
                    $scope.triples = [];
                    $scope.solutions = false;

                    let promises = [];

                    function parse_chunk(chunk) {
                        let quads = chunk.split('\n');
                        quads = quads.filter(function (q) {
                            return q != undefined && q.length > 0
                        });
                        quads.forEach(function (quad) {
                            let promise = $q(function (resolve, reject) {
                                let triple = quad.split('·').slice(1);
                                if (triple.length == 3) {
                                    let subject = createNode($scope.store.rdf, triple[0]);
                                    let predicate = createNode($scope.store.rdf, triple[1]);
                                    let object = createNode($scope.store.rdf, triple[2]);
                                    $scope.graph.add($scope.store.rdf.createTriple(subject, predicate, object));
                                    $scope.triples.push(triple);
                                    resolve(triple);
                                } else {
                                    resolve(triple);
                                }
                            });
                            promises.push(promise);
                        });
                    }

                    rdfstore.create(function (err, store) {
                        $scope.store = store;
                        $scope.graph = store.rdf.createGraph();

                        store.registerDefaultProfileNamespaces();

                        $http.get('http://localhost:5000/prefixes').success(function (data) {
                            console.log(data);

                            for (let property in data) {
                                if (data.hasOwnProperty(property)) {
                                    $scope.store.rdf.setPrefix(property, data[property]);
                                }
                            }

                            let lastLoaded = 0;
                            let preFill = '';
                            $http({
                                url: 'http://localhost:5000/fragment?query=' + encodeURIComponent($scope.query),
                                headers: {'Accept': 'application/agora-quad'},
                                eventHandlers: {
                                    progress: function (event) {
                                        if (event.loaded != undefined) {
                                            let nlIndex = event.target.response.lastIndexOf('\n');
                                            let chunk = preFill + event.target.response.substring(lastLoaded, nlIndex);
                                            lastLoaded = nlIndex;
                                            preFill = event.target.response.substring(nlIndex);
                                            promises.push(parse_chunk(chunk));
                                        }
                                    }
                                }
                            }).success(function (d) {
                                $q.all(promises).then(function (res) {
                                    console.log(res);
                                    let scope = $scope;
                                    $scope.store.insert($scope.graph, function () {
                                        $scope.store.execute($scope.query, function (err, results) {
                                            let vars = [];
                                            let bindings = [];
                                            if (results != undefined) {
                                                bindings = results.map(function (r) {
                                                    let r_prime = {};
                                                    for (let v in r) {
                                                        if (r.hasOwnProperty(v)) {
                                                            if (vars.indexOf(v) < 0) {
                                                                vars.push(v);
                                                            }
                                                            let value = null;
                                                            if (r[v] != null) {
                                                                value = r[v].value;
                                                            }
                                                            r_prime[v] = {
                                                                value: value,
                                                                type: r[v].token
                                                            }
                                                        }
                                                    }
                                                    return r_prime;
                                                });
                                            }

                                            let sparql_result = {head: {vars: vars}, results: {bindings: bindings}};
                                            console.log(sparql_result);
                                            d3sparql.htmltable(sparql_result, config);
                                            $timeout(function () {
                                                scope.solutions = true;
                                                scope.store.close();
                                            }, 0);
                                        });
                                    });
                                });
                            });
                        });
                    });
                };
            }]);
}());