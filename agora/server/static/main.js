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

// function resize()
// {
//     let heights = window.innerHeight;
//     document.getElementById("solutions").style.height = heights + "px";
// }

// window.onresize = function() {
//     resize();
// };


function textAreaAdjust(o) {
    o.style.height = "1px";
    o.style.height = (10 + o.scrollHeight) + "px";
}

let scope = undefined;

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
                // resize();
                scope = $scope;
                $scope.solutions = undefined;
                $scope.fragment = undefined;
                $scope.query = 'PREFIX wot: <http://www.wot.org#> \
SELECT * WHERE {?s rdfs:label ?l ; wot:hasLatestEntry [ wot:value ?v ] }';
                console.log('hello from SPARQL!');
                $scope.triples = [];
                $scope.results = [];
                $scope.vars = [];
                $scope.predicateMap = {};
                $scope.predicates = [];
                $scope.onSolutionsRefresh = false;
                $scope.onFragmentRefresh = false;
                $scope.ntriples = 0;

                $scope.refreshSolutions = function() {
                    if (!$scope.onSolutionsRefresh) {
                        $scope.onSolutionsRefresh = true;
                        $scope.solutionsRefreshTimer = $timeout(function () {
                            $scope.solutions = true;
                            $scope.onSolutionsRefresh = false;
                            // console.log('refreshing...');
                        }, 1);
                    }
                };

                $scope.refreshFragment = function() {
                    if (!$scope.onFragmentRefresh) {
                        $scope.onFragmentRefresh = true;
                        $scope.fragmentRefreshTimer = $timeout(function () {
                            $scope.fragment = true;
                            $scope.onFragmentRefresh = false;
                            // console.log('refreshing fragment...');
                        }, 1);
                    }
                };

                $scope.isURI = function(t) {
                    return t[0] == '<';
                };

                $scope.runQuery = function () {
                    $scope.solutions = false;
                    $scope.results = [];
                    $scope.vars = [];

                    oboe({
                        url: 'http://localhost:5000/sparql?query=' + encodeURIComponent($scope.query),
                        headers: {
                            'Accept': 'application/sparql-results+json'
                        }
                    }).node(
                        'vars.*', function (v) {
                            scope.vars.push(v);
                        }
                    ).node(
                        'bindings.*', function (r) {
                            $scope.results.push(r);
                            if ($scope.vars.length == 0) {
                                this.abort();
                                $scope.solutionsRefreshTimer.cancel();
                            } else {
                                $scope.refreshSolutions();
                            }
                        }
                    ).done(function () {
                        if ($scope.solutionsRefreshTimer != undefined) {
                            $scope.solutionsRefreshTimer.cancel();
                        }
                    });
                };

                $scope.predicateSO = function (p) {
                    return $scope.predicateMap[p];
                };

                $scope.canceller = $q.defer();

                $scope.getFragment = function () {
                    $scope.canceller.resolve();
                    $scope.canceller = $q.defer();
                    $scope.triples = [];
                    $scope.predicates = [];
                    $scope.predicateMap = {};
                    $scope.fragment = false;
                    $scope.ntriples = 0;

                    function parse_chunk(chunk) {
                        let quads = chunk.split('\n');
                        quads = quads.filter(function (q) {
                            return q != undefined && q.length > 0
                        });
                        quads.forEach(function (quad) {
                            let promise = $q(function (resolve, reject) {
                                let triple = quad.split('·').slice(1);
                                if (triple.length == 3) {
                                    $scope.ntriples++;
                                    $scope.triples.push(triple);
                                    if ($scope.predicateMap[triple[1]] == undefined) {
                                        $scope.predicateMap[triple[1]] = [];
                                        $scope.predicates.push(triple[1]);
                                    }
                                    $scope.predicateMap[triple[1]].push([triple[0], triple[2]]);
                                    resolve(triple);
                                } else {
                                    resolve(triple);
                                }
                            });
                        });
                        $scope.refreshFragment();
                    }

                    let lastLoaded = 0;
                    let preFill = '';
                    $http({
                        url: 'http://localhost:5000/fragment?query=' + encodeURIComponent($scope.query),
                        headers: {'Accept': 'application/agora-quad-min'},
                        eventHandlers: {
                            progress: function (event) {
                                if (event.loaded != undefined) {
                                    let nlIndex = event.target.response.lastIndexOf('\n');
                                    let chunk = preFill + event.target.response.substring(lastLoaded, nlIndex);
                                    lastLoaded = nlIndex;
                                    preFill = event.target.response.substring(nlIndex);
                                    parse_chunk(chunk);
                                }
                            }
                        },
                        timeout: $scope.canceller.promise
                    }).success(function (d) {
                    });
                };
            }]);
}());