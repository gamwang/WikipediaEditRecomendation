from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
import numpy as np
import json
from sets import Set

N_NEIGHBORS = 15

def get_data(count):
    f = open('./articles_with_categories.json', 'r')
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
    ids, intros, labels = get_data(1000)
    categories = list(Set(labels))
    labels = map(lambda x: categories.index(x), labels)

    train_ids, train_intros, train_labels = ids[:800], intros[:800], labels[:800]
    test_ids, test_intros, test_labels = ids[800:], intros[800:], labels[800:]

    featurizer = TfidfVectorizer(analyzer='word', stop_words='english',
            ngram_range=(1,3), min_df=0.05)
    X_train = featurizer.fit_transform(train_intros)
    X_test = featurizer.transform(test_intros)

    classifier = KNeighborsClassifier(N_NEIGHBORS, weights='distance')
    classifier.fit(X_train, train_labels)

    pred = classifier.predict(X_test)
    score = accuracy_score(test_labels, pred)
    print score


if __name__ == "__main__":
    main()
