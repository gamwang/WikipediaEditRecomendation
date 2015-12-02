import re, grequests, json
from sets import Set


#pageids is a sample of wiki page ids from a given cluster. should be around ~ 10 ids
#should write urls to output file
def get_urls_from_pageids(pageids):
    request_set = Set()
    with open('urls.txt', 'a+') as f:
        urls = [
                "http://en.wikipedia.org/w/api.php?action=query&prop=info&inprop=url&format=json",
                "http://en.wikipedia.org/w/api.php?action=query&prop=info&inprop=url&format=json",
                "http://en.wikipedia.org/w/api.php?action=query&prop=info&inprop=url&format=json",
                "http://en.wikipedia.org/w/api.php?action=query&prop=info&inprop=url&format=json",
                "http://en.wikipedia.org/w/api.php?action=query&prop=info&inprop=url&format=json",
                "http://en.wikipedia.org/w/api.php?action=query&prop=info&inprop=url&format=json",
                "http://en.wikipedia.org/w/api.php?action=query&prop=info&inprop=url&format=json",
                "http://en.wikipedia.org/w/api.php?action=query&prop=info&inprop=url&format=json",
                "http://en.wikipedia.org/w/api.php?action=query&prop=info&inprop=url&format=json"
                ]
        
        #add pageid param to request url
        for i in range(len(urls)):
            pageid_param = "&pageids=" + pageids[i]
            urls[i] = urls[i] + pageid_param

        rs = grequests.map((grequests.get(u) for u in urls))
        for r in rs:
            print(r)
            r_json = r.json()
            page_info = r_json['query']['pages']
            fullurl = page_info[page_info.keys()[0]]['fullurl']
            #write url to some output file
            f.write(fullurl)
            f.write('\n')
pageids = ["10289104", "1098410", "28083728", "14636948", "7935893", "46877779", "47691713", "31333430", "21418402", "32453694"]
get_urls_from_pageids(pageids)

