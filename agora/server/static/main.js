/**
 * Created by fserena on 28/11/16.
 */

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
        .controller('SPARQLController', ['$scope', '$log', '$http', '$timeout', '$q', '$location',
            function ($scope, $log, $http, $timeout, $q, $location) {
                // resize();
                scope = $scope;
                $scope.solutions = undefined;
                $scope.fragment = undefined;
                $scope.query = '';
                console.log('hello from SPARQL!');
                $scope.triples = [];
                $scope.results = [];
                $scope.vars = [];
                $scope.predicateMap = {};
                $scope.predicates = [];
                $scope.onSolutionsRefresh = false;
                $scope.onFragmentRefresh = false;
                $scope.ntriples = 0;
                $scope.queryRunning = false;
                $scope.fragRunning = false;
                $scope.onError = false;

                $scope.refreshSolutions = function () {
                    if (!$scope.onSolutionsRefresh) {
                        $scope.onSolutionsRefresh = true;
                        $scope.solutionsRefreshTimer = $timeout(function () {
                            $scope.solutions = true;
                            $scope.onSolutionsRefresh = false;
                            // console.log('refreshing...');
                        }, 1);
                    }
                };

                $scope.refreshFragment = function () {
                    if (!$scope.onFragmentRefresh) {
                        $scope.onFragmentRefresh = true;
                        $scope.fragmentRefreshTimer = $timeout(function () {
                            $scope.fragment = true;
                            $scope.onFragmentRefresh = false;
                            // console.log('refreshing fragment...');
                        }, 1);
                    }
                };

                $scope.isURI = function (t) {
                    return t[0] == '<';
                };

                $scope.runQuery = function () {
                    $scope.solutions = false;
                    $scope.queryRunning = true;
                    $scope.results = [];
                    $scope.vars = [];

                    let qObj = $location.search();
                    let qArgs = '';
                    for (let k in qObj) {
                        if (qObj.hasOwnProperty(k)) {
                            qArgs += '&' + k + '=' + qObj[k];
                        }
                    }

                    let baseUrl = $location.protocol() + '://' + $location.host();
                    if ($location.port() !== undefined && $location.port() !== 80) {
                        baseUrl += ':' + $location.port();
                    }

                    baseUrl += $location.path();

                    oboe({
                        url: baseUrl + 'sparql?query=' + encodeURIComponent($scope.query) + qArgs,
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
                        try {
                            if ($scope.solutionsRefreshTimer !== undefined) {
                                $scope.solutionsRefreshTimer.cancel();
                            }
                        } catch (e) {
                        }
                        $scope.queryRunning = false;
                        $scope.$apply();
                    }).fail(function () {
                        $scope.queryRunning = false;
                        $scope.onError = true;
                        $scope.$apply();
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
                    $scope.fragRunning = true;

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

                    let qObj = $location.search();
                    let qArgs = '';
                    for (let k in qObj) {
                        if (qObj.hasOwnProperty(k)) {
                            qArgs += '&' + k + '=' + qObj[k];
                        }
                    }

                    let baseUrl = $location.protocol() + '://' + $location.host();
                    if ($location.port() !== undefined && $location.port() !== 80) {
                        baseUrl += ':' + $location.port();
                    }

                    baseUrl += $location.path();

                    let lastLoaded = 0;
                    let preFill = '';
                    $http({
                        url: baseUrl + 'fragment?query=' + encodeURIComponent($scope.query) + qArgs,
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
                        $scope.fragRunning = false;
                    }).error(function(d) {
                        $scope.fragRunning = false;
                        $scope.onError = true;
                    });
                };
            }]);
}());