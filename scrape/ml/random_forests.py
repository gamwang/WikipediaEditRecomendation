from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
import numpy as np
import requests
import json
from sets import Set

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

    #test data
    test_ids = ids[split_index:]
    test_intros = intros[split_index:]
    test_labels = labels[split_index:]

    #featurizer = TfidfVectorizer(analyzer='word', stop_words='english',
    #        ngram_range=(1,3), min_df=0.03)
    featurizer = CountVectorizer(analyzer='word', stop_words='english',
            ngram_range=(1,3), min_df=0.03)
    X_train = featurizer.fit_transform(train_intros)
    X_test = featurizer.transform(test_intros)

    classifier = RandomForestClassifier()
    classifier.fit(X_train, train_labels)

    pred = classifier.predict(X_test)
    score = accuracy_score(test_labels, pred)
    print score

if __name__ == "__main__":
    main()
