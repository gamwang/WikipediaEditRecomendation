'use strict';

var request = require('request');
var _ = require('underscore');
var fs = require('fs');
var log = fs.createWriteStream('users.txt', {'flags': 'a'});

function scrapeUserNames(prefix, cb) { 
    var url = 'https://en.wikipedia.org/w/api.php?action=query&list=allusers&aufrom=' + prefix + '&aulimit=' + 500 + '&format=json&auwitheditsonly=true';
    request(url, function (err, resp, body) {
        cb(err, resp, JSON.parse(body));
    });    
}

function scrapeUserContrib(users) {
    _.each(users, function (user) {
        var name = user.name;
        var url = 'https://en.wikipedia.org/w/api.php?action=query&list=usercontribs&ucuser=' + user.name + '&uclimit=500&ucdir=newer&format=json';
        request(url, function (err, resp, body) {
            if (err) {
                console.log('error');
                console.log(url);
                console.log(err);
            } else {
                log.write(body);
                log.write('\n');
            }
        });
    });
}
for (var count = 0; count < 1; count += 1) {
    var possible = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
    var prefix = '';
    for (var i = 0; i < 1; i += 1) {
        prefix += possible.charAt(Math.floor(Math.random() * possible.length));
    }

    scrapeUserNames(prefix, function (err, resp, body) {
        scrapeUserContrib(body.query.allusers);
    });
}
