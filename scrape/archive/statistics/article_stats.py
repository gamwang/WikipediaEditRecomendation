import json
from sets import Set
from collections import Counter


def main():
    total = 0.0
    articles = 0
    articleset = Set()

    counts = []

    with open('articles.json', 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
            except ValueError:
                continue

            pageinfo = data['pages']
            for key in pageinfo.keys():
                if key not in articleset:
                    articleset.add(key)
                    extract = pageinfo[key]['extract']
                    temp = extract.split(' ')
                    if len(temp) < 1:
                        continue
                    articles += 1
                    total += len(temp)

                counts.append(len(temp))

    average = total/articles
    counts = Counter(counts)
    print average
    print counts

main()