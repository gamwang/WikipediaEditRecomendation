#! /usr/bin/env python
import re, requests, json
import sys, os

"""
Usage: ./categories.py [filename] [directory]

[filename] should contain the categories that you want to query, one per line.
    Including the leading "Category:" tag in the file is optional.

The resulting json files are written to [directory]. If [directory] exists, an
error message is printed.
"""

def main():
    if len(sys.argv) != 3:
        print "Usage: ./categories.py [filename] [directory]"
        sys.exit(1)

    category_file = sys.argv[1]
    directory = sys.argv[2]
    try:
        os.mkdir(directory)
    except OSError:
        print "Error directory already exists."
        sys.exit(1)

    try:
        with open(category_file, 'r') as cf:
            for category in cf:
                category = category.strip()
                if "Category:" not in category:
                    category = "Category:" + category.strip()
                category_request(category, directory)
    except IOError:
        print "Error: cannot find the input file or read the input data."
        os.rmdir(directory)
        sys.exit(1)


def category_request(category, directory):
    cat_string = category.split(":")[1]
    filename = directory + "/" +cat_string+".json"
    try:
        with open(filename, 'a+') as f:
            url ="https://en.wikipedia.org/w/api.php?action=query&list=categorymembers&cmtitle="
            url = url + category
            url = url + "&cmlimit=500&cmtype=page&format=json"
            r = requests.get(url)
            r_json = r.json()
            json.dump(r_json, f)
    except ValueError:
        print "Error: reponse is not in json format."
        sys.exit(1)

if __name__ == "__main__":
    main()
