/** Get extracts from list of Wikipedia page IDs. */

var request = require('request');
var async = require('async');
var fs = require('fs');

var OUT_FILE = "articles_with_categories.json";
var FOLDER = "labeled/";
var CATEGORIES = ["Games", "Media", "Medicine", "Music", "Politics", "Sports"];
var API_URL = "https://en.wikipedia.org/w/api.php?action=query&format=json&prop=extracts&exintro&explaintext&pageids=";

function filterData(data, id) {
    return data['query']['pages'][id]["extract"];
}

CATEGORIES.forEach(function(category) {
    var dir = FOLDER + category + "/";

    fs.readdir(dir, function (err, files) {
        files.forEach(function (file) {
            fs.readFile(dir + file, 'utf-8', function (err, contents) {
                var obj = JSON.parse(contents);
                var pages = obj['query']['categorymembers'];
                var pageIds = pages.map(function (page) { return page['pageid']; });

                writeJSON(category, pageIds);
            });
        });
    });
});

function writeJSON(category, pageIds) {
    console.log("Working on category: " + category);

    async.eachSeries(pageIds, function (pageId, cb) {
        var url = API_URL + pageId;

        console.log(url);

        request(url, function (err, res, body) {
            try {
                var data = JSON.parse(body);
                var filteredData = data['query'];
                filteredData['pages'][pageId]['category'] = category;

                fs.appendFileSync(OUT_FILE, JSON.stringify(filteredData) + "\n", 'utf8');
            } catch (e) {
            }
            cb();
        });
    }, function done() {
        console.log("Done!");
    });
}
