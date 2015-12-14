import json
import matplotlib.pyplot as plt
from sets import Set
from collections import Counter


def main():
    total = 0.0
    users = 0
    userset = Set()

    counts = []

    for i in range(10):
        filename = str(i) + ".json"
        with open(filename, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                except ValueError:
                    continue

                if 'query' in data:
                    contribs = data['query']['usercontribs']
                    if len(contribs) < 1:
                        continue
                    userid = contribs[0]['userid']
                    if userid not in userset:
                        users += 1
                        if len(contribs) == 500:
                            continue
                        total += len(contribs)
                        userset.add(userid)

                        counts.append(len(contribs))

    average = total/users
    print average
    plt.hist(counts)
    plt.title("Contributor Histogram")
    plt.xlabel("Number of Contributions")
    plt.ylabel("Number of Contributors")
    plt.show()


main()
