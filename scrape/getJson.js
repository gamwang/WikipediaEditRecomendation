var fs = require('fs');
var _ = require('underscore');
var log = fs.createWriteStream('./data/json/contributions2.json');
var async = require('async');

var sc = require('./scrape');

fs.readFile('./data/contribs/contributions2.txt', function (err, resp) {
    var users = resp.toString().split('\n');
    // getting rid of empy string
    users.pop();
    _.each(users, function (user) {
        user = JSON.parse(user);
        var contribs = user.query.usercontribs;
        var output = {};
        console.log(contribs);
        if (contribs.length > 0) {
            output.uid = contribs[0].userid;
            output.categories = [];
            var fns = [];
            
            _.each(contribs, function (contrib) {
                fns.push(function (done) {
                    var title = contrib.title;
                    sc.getCategory(title, function (err, resp, body) {
                        console.log(body);
                        try {
                            JSON.parse(body);
                        } catch (e) {
                            done();
                            return;
                        }
                        if (err) {
                            done();
                            return;
                        }
                        var data = JSON.parse(body);
                        if (!data.query) {
                            done();
                            return;
                        }
                        var categories = data.query.pages; 
                        output.categories.push(categories);
                        done();
                    });
                });
            });
            async.parallel(fns, function () {
                log.write(JSON.stringify(output));
                log.write('\n');
            });
        }
    });
    
});
