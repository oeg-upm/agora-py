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


let scope = undefined;
let mymap = undefined;
let yasqe = undefined;


let geojsonMarkerOptions = {
    radius: 8,
    fillColor: "#509dc8",
    color: "#000",
    weight: 1,
    opacity: 1,
    fillOpacity: 0.8
};

let circlePoint = function (feature, latlng) {
    return L.circleMarker(latlng, geojsonMarkerOptions);
};

let popUpName = function (feature, layer) {
    // does this feature have a property named popupContent?
    if (Object.keys(feature.properties).length) {
        let message = "<div>";
        Object.keys(feature.properties).map(function (key, index) {
            message += "<strong>" + key + "</strong>: " + feature.properties[key] + "<br>";
        });
        message += '</div>';
        layer.bindPopup(message);
    }
};

(function () {

    'use strict';

    angular.module('AgoraApp', ['ngRoute', 'ngAnimate'])
        .config(['$routeProvider', '$locationProvider',
            function ($routeProvider, $locationProvider) {
                let path = window.location.pathname;
                if (!path.endsWith('/')) {
                    path = path + '/';
                }

                $routeProvider
                    .otherwise({
                        templateUrl: path + 'sparql.html',
                        controller: 'SPARQLController',
                        controllerAs: 'sparql'
                    });

                // $locationProvider.html5Mode(true);
            }])
        .controller('SPARQLController', ['$scope', '$log', '$http', '$timeout', '$q', '$location', '$window',
            function ($scope, $log, $http, $timeout, $q, $location, $window) {
                mymap = L.map('mapid').setView([51.505, -0.09], 13);

                L.tileLayer('https://api.tiles.mapbox.com/v4/{id}/{z}/{x}/{y}.png?access_token={accessToken}', {
                    attribution: 'Map data &copy; <a href="http://openstreetmap.org">OpenStreetMap</a> contributors, <a href="http://creativecommons.org/licenses/by-sa/2.0/">CC-BY-SA</a>, Imagery © <a href="http://mapbox.com">Mapbox</a>',
                    maxZoom: 18,
                    id: 'mapbox.streets',
                    accessToken: 'pk.eyJ1IjoiZnNlcmVuYSIsImEiOiJjajU5aWUwdXYwOXFnMnF0Z3gxNzY3cHBrIn0.Ks47QtJAcy3ybvq-9NoD4w'
                }).addTo(mymap);

                scope = $scope;
                $scope.solutions = undefined;
                $scope.fragment = undefined;
                // $scope.query = '';
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
                $scope.geoSolutions = false;
                $scope.fragRunning = false;
                $scope.onError = false;
                $scope.layerGroup = L.geoJSON();
                $scope.layerGroup.addTo(mymap);
                $scope.oboe = undefined;
                $scope.requestAbortFragment = false;
                $scope.canceller = $q.defer();
                $scope.queryStoppable = false;
                $scope.fragStoppable = false;
                $scope.latLons = [];

                $scope.baseUrl = $location.protocol() + '://' + $location.host();
                if ($location.port() !== undefined && $location.port() !== 80) {
                    $scope.baseUrl += ':' + $location.port();
                }

                $scope.baseUrl += $window.location.pathname;
                if (!$scope.baseUrl.endsWith('/')) {
                    $scope.baseUrl += '/';
                }

                YASQE.defaults.sparql.showQueryButton = false;
                YASQE.defaults.sparql.endpoint = $scope.baseUrl + 'sparql';

                let customPrefixesCompleter = function (yasqe) {
                    let returnObj = {
                        isValidCompletionPosition: function () {
                            return YASQE.Autocompleters.prefixes.isValidCompletionPosition(yasqe)
                        },
                        preProcessToken: function (token) {
                            return YASQE.Autocompleters.prefixes.preProcessToken(yasqe, token)
                        },
                        postProcessToken: function (token, suggestedString) {
                            return YASQE.Autocompleters.prefixes.postProcessToken(yasqe, token, suggestedString)
                        }
                    };
                    returnObj.bulk = true;
                    returnObj.async = true;
                    returnObj.autoShow = true;
                    returnObj.get = function (token, callback) {
                        $.ajax({
                            url: $scope.baseUrl + 'prefixes',
                            headers: {Accept: "application/json"},//ask for csv. Simple, and uses less bandwidth
                            success: function (data) {
                                scope.prefixes = data;
                                let prefixes = [];
                                Object.keys(data).map(function (key, index) {
                                    prefixes.push(key + ': <' + data[key] + '>');
                                });
                                callback(prefixes);
                            }
                        });
                    };
                    return returnObj;
                };

                YASQE.registerAutocompleter('A_PrefixesCompleter', customPrefixesCompleter);


                let customPropertyCompleter = function (yasqe) {
                    //we use several functions from the regular property autocompleter (this way, we don't have to re-define code such as determining whether we are in a valid autocompletion position)
                    let returnObj = {
                        isValidCompletionPosition: function () {
                            return YASQE.Autocompleters.properties.isValidCompletionPosition(yasqe)
                        },
                        preProcessToken: function (token) {
                            return YASQE.Autocompleters.properties.preProcessToken(yasqe, token)
                        },
                        postProcessToken: function (token, suggestedString) {
                            return YASQE.Autocompleters.properties.postProcessToken(yasqe, token, suggestedString)
                        }
                    };

                    //In this case we assume the properties will fit in memory. So, turn on bulk loading, which will make autocompleting a lot faster
                    returnObj.bulk = true;
                    returnObj.async = true;

                    //and, as everything is in memory, enable autoShowing the completions
                    returnObj.autoShow = true;

                    // returnObj.persistent = "customProperties";//this will store the sparql results in the client-cache for a month.
                    returnObj.get = function (token, callback) {
                        $.ajax({
                            url: $scope.baseUrl + 'properties',
                            headers: {Accept: "application/json"},//ask for csv. Simple, and uses less bandwidth
                            success: function (data) {
                                let props = [];
                                Object.keys(scope.prefixes).map(function (key, index) {
                                    Array.from(data.properties).map(function (elm) {
                                        if (elm.indexOf(key) > -1) {
                                            props.push(elm.replace(key + ':', scope.prefixes[key]))
                                        }
                                    });
                                });
                                callback(props);
                            }
                        });
                    };
                    return returnObj;
                };
                //now register our new autocompleter
                YASQE.registerAutocompleter('customPropertyCompleter', customPropertyCompleter);

                //excellent, now do the same for the classes
                let customClassCompleter = function (yasqe) {
                    let returnObj = {
                        isValidCompletionPosition: function () {
                            return YASQE.Autocompleters.classes.isValidCompletionPosition(yasqe)
                        },
                        preProcessToken: function (token) {
                            return YASQE.Autocompleters.classes.preProcessToken(yasqe, token)
                        },
                        postProcessToken: function (token, suggestedString) {
                            return YASQE.Autocompleters.classes.postProcessToken(yasqe, token, suggestedString)
                        }
                    };
                    returnObj.bulk = true;
                    returnObj.async = true;
                    returnObj.autoShow = true;
                    returnObj.get = function (token, callback) {
                        $.ajax({
                            url: $scope.baseUrl + 'types',
                            headers: {Accept: "application/json"},//ask for csv. Simple, and uses less bandwidth
                            success: function (data) {
                                let types = [];
                                Object.keys(scope.prefixes).map(function (key, index) {
                                    Array.from(data.types).map(function (elm) {
                                        if (elm.indexOf(key) > -1) {
                                            types.push(elm.replace(key + ':', scope.prefixes[key]))
                                        }
                                    });
                                });
                                callback(types);
                            }
                        });
                    };
                    return returnObj;
                };

                YASQE.registerAutocompleter('customClassCompleter', customClassCompleter);

                //And, to make sure we don't use the other property and class autocompleters, overwrite the default enabled completers
                // YASQE.defaults.autocompleters = ['customClassCompleter', 'customPropertyCompleter', 'customPrefixesCompleter'];
                YASQE.defaults.autocompleters = $.grep(YASQE.defaults.autocompleters, function (completer) {
                    return completer !== 'prefixes'
                });
                YASQE.defaults.autocompleters = $.grep(YASQE.defaults.autocompleters, function (completer) {
                    return completer !== 'classes'
                });
                YASQE.defaults.autocompleters = $.grep(YASQE.defaults.autocompleters, function (completer) {
                    return completer !== 'properties'
                });


                yasqe = YASQE.fromTextArea(document.getElementById('query'));
                yasqe.on("change", function () {
                    YASQE.Autocompleters.prefixes.appendPrefixIfNeeded(yasqe, 'A_PrefixesCompleter');
                    scope.onError = false;
                });

                $(".fullscreenToggleBtns").css("display", "none");

                $scope.refreshSolutions = function () {
                    if (!$scope.onSolutionsRefresh) {
                        $scope.onSolutionsRefresh = true;
                        $scope.solutionsRefreshTimer = $timeout(function () {
                            $scope.solutions = true;
                            $scope.onSolutionsRefresh = false;
                            if ($scope.geoSolutions) {
                                mymap.fitBounds($scope.layerGroup.getBounds());
                            }
                        }, 1);
                    }
                };

                $scope.refreshFragment = function () {
                    if (!$scope.onFragmentRefresh) {
                        $scope.onFragmentRefresh = true;
                        $scope.fragmentRefreshTimer = $timeout(function () {
                            $scope.fragment = true;
                            $scope.onFragmentRefresh = false;
                        }, 1);
                    }
                };

                $scope.isURI = function (t) {
                    return t[0] === '<';
                };

                $scope.openLink = function (link) {
                    if ($scope.isURI(link)) {
                        $window.open(link.replace('<', '').replace('>', ''), '_blank');
                    }
                };

                $scope.abortQuery = function () {
                    if ($scope.oboe !== undefined) {
                        console.log('aborting query...');
                        $scope.oboe.abort();
                        $scope.queryRunning = false;
                        $scope.queryStoppable = false;
                    }
                };

                $scope.runQuery = function () {
                    $scope.layerGroup.clearLayers();
                    $scope.solutions = false;
                    $scope.queryStoppable = false;
                    $scope.queryRunning = true;
                    $scope.geoSolutions = false;
                    $scope.latLons = [];
                    $scope.results = [];
                    $scope.vars = [];
                    $scope.onError = false;

                    $timeout(function () {
                        if ($scope.queryRunning) {
                            $scope.queryStoppable = true;
                        }
                    }, 1000);

                    let qArgs = '&' + window.location.search.substring(1);

                    let oboeObj = oboe({
                        url: $scope.baseUrl + 'sparql?query=' + encodeURIComponent(yasqe.getValue()) + qArgs,
                        headers: {
                            'Accept': 'application/sparql-results+json'
                        }
                    });

                    $scope.oboe = oboeObj;

                    oboeObj.node(
                        'vars.*', function (v) {
                            scope.vars.push(v);
                        }
                    ).node(
                        'bindings.*', function (r) {
                            $scope.results.push(r);
                            let geometries = [];
                            let props = {};
                            Object.keys(r).map(function (key, index) {
                                try {
                                    let geometry = Terraformer.WKT.parse(r[key].value);
                                    geometries.push(geometry);
                                    $scope.geoSolutions = true;
                                } catch (Exception) {
                                    props[key] = r[key].value;
                                }
                            });

                            geometries.forEach(function(geom) {
                                let lgeo = L.geoJSON({
                                            "type": "Feature",
                                            "properties": props,
                                            "geometry": geom
                                        },
                                        {
                                            pointToLayer: circlePoint,
                                            onEachFeature: popUpName
                                        });

                                $scope.layerGroup.addLayer(lgeo);
                            });


                            if (r.lat !== undefined && r.lon !== undefined) {
                                let latLonStr = String(r.lat.value) + String(r.lon.value);

                                if ($scope.latLons.indexOf(latLonStr) < 0) {
                                    $scope.latLons.push(latLonStr);
                                    let lgeo = L.geoJSON({
                                            type: "Feature",
                                            properties: props,
                                            geometry: {
                                                "type": "Point",
                                                "coordinates": [r.lon.value, r.lat.value]
                                            }
                                        },
                                        {
                                            pointToLayer: circlePoint,
                                            onEachFeature: popUpName
                                        });
                                    $scope.layerGroup.addLayer(lgeo);
                                    $scope.geoSolutions = true;
                                }
                            }

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
                        $scope.queryStoppable = false;
                        if ($scope.geoSolutions) {
                            mymap.fitBounds($scope.layerGroup.getBounds());
                        }
                        oboeObj.abort();
                        $scope.$apply();
                    }).fail(function () {
                        $scope.queryRunning = false;
                        $scope.queryStoppable = false;
                        $scope.geoSolutions = false;
                        $scope.onError = true;
                        oboeObj.abort();
                        $scope.$apply();
                    });
                };

                $scope.predicateSO = function (p) {
                    return $scope.predicateMap[p];
                };

                $scope.abortFragment = function () {
                    $scope.requestAbortFragment = true;
                    $scope.canceller.resolve();
                    $scope.fragRunning = false;
                    $scope.fragStoppable = false;
                };

                $scope.getFragment = function () {
                    $scope.canceller.resolve();
                    $scope.canceller = $q.defer();
                    $scope.triples = [];
                    $scope.predicates = [];
                    $scope.predicateMap = {};
                    $scope.fragment = false;
                    $scope.ntriples = 0;
                    $scope.fragRunning = true;
                    $scope.fragStoppable = false;
                    $scope.onError = false;
                    $scope.requestAbortFragment = false;

                    $timeout(function () {
                        if ($scope.fragRunning) {
                            $scope.fragStoppable = true;
                        }
                    }, 1000);

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
                    let qArgs = '&' + window.location.search.substring(1);

                    let lastLoaded = 0;
                    let preFill = '';
                    $http({
                        url: $scope.baseUrl + 'fragment?query=' + encodeURIComponent(yasqe.getValue()) + qArgs,
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
                        $scope.fragStoppable = false;
                    }).error(function (d) {
                        $scope.fragRunning = false;
                        $scope.fragStoppable = false;
                        if (!$scope.requestAbortFragment) {
                            $scope.onError = true;
                        }
                    });
                };
            }]);
}());