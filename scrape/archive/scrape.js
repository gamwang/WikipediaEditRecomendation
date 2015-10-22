'use strict';

var request = require('request');
var _ = require('underscore');
var fs = require('fs');

module.exports = {
    scrapeUserNames: function (prefix, cb) { 
        var url = 'https://en.wikipedia.org/w/api.php?action=query&list=allusers&aufrom=' + prefix + '&aulimit=' + 500 + '&format=json&auwitheditsonly=true';
        request(url, function (err, resp, body) {
            cb(err, resp, JSON.parse(body));
        });    
    },

    scrapeUserContrib: function (users) {
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
    },

    getCategory: function (title, cb) {
        var url = 'https://en.wikipedia.org/w/api.php?action=query&generator=categories&titles=' + title + '&prop=info&format=json';
        request(url, cb);
    }

}

var cmdline = process.argv;
var log = fs.createWriteStream(cmdline[3], {'flags': 'a'});
if (cmdline[2] == '-con') {
    for (var count = 0; count < 1; count += 1) {
        var possible = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
        var prefix = '';
        for (var i = 0; i < 1; i += 1) {
            prefix += possible.charAt(Math.floor(Math.random() * possible.length));
        }
        module.exports.scrapeUserNames(prefix, function (err, resp, body) {
            module.exports.scrapeUserContrib(body.query.allusers);
        });
    }
} else if (cmdline[2] == '-cat') {
    module.exports.getCategory(cmdline[4], function (err, resp, body) {
        log.write(body);
    });
}
