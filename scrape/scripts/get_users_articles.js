/** Gets random users and creates a dictinoary of the form:
 *  user -> (title, excerpt).
 *
 *  Usage: node get_users_articles.js [numUsers] [outfile] */

var request = require('request');
var async = require('async');
var fs = require('fs');

var out = process.argv[3] || 'out.json';
var numUsers = process.argv[2] || 500;
var numCharsInPrefix = 3;
var maxPagesPerUser = 100;

var allUsers = [];
var result = {};

function done() {
    return Object.keys(allUsers).length > numUsers;
}

function getRandSequence(numChars) {
    var text = '';
    var alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";

    for (var i = 0; i < numChars; i++) {
        text += alphabet.charAt(Math.floor(Math.random() * alphabet.length));
    }

    return text;
}

async.until(done, function (cb) {
    var prefix = getRandSequence(numCharsInPrefix);
    var url = "https://en.wikipedia.org/w/api.php?action=query&list=allusers&auprefix=" + prefix + "&aulimit=500&format=json&auwitheditsonly=true";

    console.log("Testing prefix: " + prefix);

    request(url, function (err, res, body) {
        var data = JSON.parse(body);
        var users = data.query.allusers;

        users.forEach(function (user) {
            allUsers.push(user.name);
        });

        cb();
    });

}, function () {
    console.log("Done getting users.");

    async.eachLimit(allUsers, 20, function (username, cb) {
        var url = "https://en.wikipedia.org/w/api.php?action=query&list=usercontribs&ucuser=" + username + "&uclimit=500&ucdir=newer&format=json";

        console.log("Getting contributions for " + username);

        request(url, function (err, res, body) {
            var data = {};
            try {
                data = JSON.parse(body);
            } catch (e) {
                console.log("Failed: " + url);
                cb();
                return;
            }
            var contribs = data.query.usercontribs;

            if (contribs.length == 0) {
                cb();
                return;
            }

            var resultContribs = [];

            async.eachLimit(contribs, 20, function (contrib, innerCb) {
                if (resultContribs.length > maxPagesPerUser || resultContribs.length >= contribs.legnth) {
                    innerCb();
                    return;
                }

                console.log("Getting extracts for " + contrib.title);

                // Get extracts
                var pageId = contrib.pageid;
                var extractUrl = "https://en.wikipedia.org/w/api.php?action=query&prop=extracts&format=json&exintro=&explaintext=&pageids=" + pageId;

                request(extractUrl, function (err, res, body) {
                    try {
                        var data = JSON.parse(body);
                        var extract = data.query.pages[pageId].extract;

                        resultContribs.push({
                            title: contrib.title,
                            extract: extract
                        });
                    } catch (e) {}

                    innerCb();
                });
            }, function () {
                // Done getting extracts
                console.log("Done getting extracts for " + username);
                result[username] = resultContribs;
                cb();
            });

        });
    }, function () {
        fs.writeFileSync(out, JSON.stringify(result, null, 4));
        console.log("Done, wrote JSON to " + out);
    });
})