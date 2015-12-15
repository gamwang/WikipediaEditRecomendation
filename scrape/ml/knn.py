from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
from math import sqrt, ceil
import numpy as np
import requests
import json
import random
from sets import Set
from collections import Counter
import sys

def get_data(count):
    f = open('../articles_with_categories.json', 'r')
    objs = f.read().split('\n')
    ids = []
    articles = []
    labels = []
    i = 0
    for obj in objs:
        if i > count:
            break
        if obj is None or len(obj.strip()) == 0:
            continue
        json_obj = json.loads(obj)
        page_info = json_obj['pages']
        info = page_info[page_info.keys()[0]]
        pid = info['pageid']
        extract = info['extract']
        labels.append(info['category'])
        ids.append(pid)
        articles.append(extract)
        i += 1
    return ids, articles, labels

def get_user_data():
    titles = []
    extracts = []
    if len(sys.argv) != 2:
        print "Please provide one input json filename."
        sys.exit(1)
    filename = sys.argv[1]
    with open(filename, 'r') as f:
        data = json.loads(f.read())
        for key in data.keys():
            edits = data[key]
            for article in edits:
                if len(article['extract']) == 0:
                    continue
                if ":" in article['title']:
                    continue
                titles.append(article['title'])
                extracts.append(article['extract'])
    return titles, extracts

def main():
    N = 1500
    split = 0.8
    split_index = int(N * split)
    ids, intros, labels = get_data(2000)
    categories = list(Set(labels))
    labels = map(lambda x: categories.index(x), labels)

    #train data
    train_ids = ids[:split_index]
    train_intros = intros[:split_index]
    train_labels = labels[:split_index]

    #randomly shuffle around test data
    temp = list(zip(train_intros, train_labels))
    random.shuffle(temp)
    train_intros, train_labels = zip(*temp)

    #test data
    test_ids = ids[split_index:]
    test_intros = intros[split_index:]
    test_labels = labels[split_index:]

    """
    featurizer = TfidfVectorizer(analyzer='word', stop_words='english',
            ngram_range=(1,3), min_df=0.03)
    """
    featurizer = CountVectorizer(analyzer='word', stop_words='english', ngram_range=(1,3), min_df=0.03)
    X_train = featurizer.fit_transform(train_intros)
    N_NEIGHBORS = int(ceil(sqrt(X_train.shape[1])))
    X_test = featurizer.transform(test_intros)

    classifier = KNeighborsClassifier(N_NEIGHBORS, weights='distance')
    classifier.fit(X_train, train_labels)

    pred = classifier.predict(X_test)
    score = accuracy_score(test_labels, pred)
    print score
    title, extracts =  get_user_data()
    extracts = featurizer.transform(extracts)
    suggestions = classifier.predict(extracts)
    suggestions = Counter(suggestions)
    print categories[suggestions.most_common(1)[0][0]]

if __name__ == "__main__":
    main()
