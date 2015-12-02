import re, grequests, json
from sets import Set

def batch_requests():
    i = 0
    request_set = Set()
    with open('temp.txt', 'a+') as f:
        while(i <= 10000):
            urls = [
                    "https://en.wikipedia.org/w/api.php?action=query&format=json&prop=extracts&exintro&explaintext&generator=random",
                    "https://en.wikipedia.org/w/api.php?action=query&format=json&prop=extracts&exintro&explaintext&generator=random",
                    "https://en.wikipedia.org/w/api.php?action=query&format=json&prop=extracts&exintro&explaintext&generator=random",
                    "https://en.wikipedia.org/w/api.php?action=query&format=json&prop=extracts&exintro&explaintext&generator=random",
                    "https://en.wikipedia.org/w/api.php?action=query&format=json&prop=extracts&exintro&explaintext&generator=random",
                    "https://en.wikipedia.org/w/api.php?action=query&format=json&prop=extracts&exintro&explaintext&generator=random",
                    "https://en.wikipedia.org/w/api.php?action=query&format=json&prop=extracts&exintro&explaintext&generator=random",
                    "https://en.wikipedia.org/w/api.php?action=query&format=json&prop=extracts&exintro&explaintext&generator=random",
                    "https://en.wikipedia.org/w/api.php?action=query&format=json&prop=extracts&exintro&explaintext&generator=random",
                    "https://en.wikipedia.org/w/api.php?action=query&format=json&prop=extracts&exintro&explaintext&generator=random",
                    "https://en.wikipedia.org/w/api.php?action=query&format=json&prop=extracts&exintro&explaintext&generator=random",
                    "https://en.wikipedia.org/w/api.php?action=query&format=json&prop=extracts&exintro&explaintext&generator=random",
                    "https://en.wikipedia.org/w/api.php?action=query&format=json&prop=extracts&exintro&explaintext&generator=random",
                    "https://en.wikipedia.org/w/api.php?action=query&format=json&prop=extracts&exintro&explaintext&generator=random",
                    "https://en.wikipedia.org/w/api.php?action=query&format=json&prop=extracts&exintro&explaintext&generator=random",
                    "https://en.wikipedia.org/w/api.php?action=query&format=json&prop=extracts&exintro&explaintext&generator=random",
                    "https://en.wikipedia.org/w/api.php?action=query&format=json&prop=extracts&exintro&explaintext&generator=random",
                    "https://en.wikipedia.org/w/api.php?action=query&format=json&prop=extracts&exintro&explaintext&generator=random",
                    "https://en.wikipedia.org/w/api.php?action=query&format=json&prop=extracts&exintro&explaintext&generator=random",
                    "https://en.wikipedia.org/w/api.php?action=query&format=json&prop=extracts&exintro&explaintext&generator=random"]
            rs = grequests.map((grequests.get(u) for u in urls))
            for r in rs:
                r_json = r.json()
                page_info = r_json['query']['pages']
                title = page_info[page_info.keys()[0]]['title']
                matched = re.match("^(.*:)|List", title)
                if not matched:
                    if (len(page_info[page_info.keys()[0]]['extract']) > 0):
                        i += 1
                        page_id = int(page_info[page_info.keys()[0]]['pageid'])
                        if (page_id not in request_set):
                            request_set.add(page_id)
                            json.dump(r_json['query'], f)
                            f.write('\n')
batch_requests()
