from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
import simplejson as json
from matplotlib import pyplot as plt
import numpy as np
import time
import operator
import math
import enchant
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score
import matplotlib.cm as cm
import os

THRESHHOLD = 7
testing = True
WORD_CHECKER = enchant.Dict("en_US")

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

def featurize(data):
    ngram_vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 3), min_df=1)
    counts = ngram_vectorizer.fit_transform(data)
    f_names = ngram_vectorizer.get_feature_names() 
    f_vals = counts.toarray().astype(int) 
    return f_names, f_vals 

def filter_features(f_names, f_vals, thresh):
    keep_list = []
    for i in range(len(f_vals)):
        for j in range(len(f_vals[i])):
            cur = f_vals[i][j]
            if cur > thresh and j not in keep_list:
                keep_list.append(j)
    out_names = []
    out_vals = []
    for i in range(len(f_vals)):
        out_vals.append(operator.itemgetter(keep_list)(f_vals[i]))
    for i in range(len(f_names)):
        if i in keep_list:
            out_names.append(f_names[i])
    return out_names, out_vals

def tfidf_weights(f_vals):
    transformer = TfidfTransformer() 
    tfidf = transformer.fit_transform(f_vals)
    return tfidf.toarray() 

def get_kmeans_estimater(data, n):
    estimater = KMeans(init='k-means++', n_clusters=n, n_init=10, max_iter=400)
    estimater.fit(data)
    return estimater

def cluster_n(data, n):
    est = get_kmeans_estimater(data, n)
    labels = est.labels_
    return est, labels

def get_output(labels, feats, k):
    feats = np.array(feats)
    div_lbls = []
    for i in range(k):
        div_lbls.append([])
    for i in range(len(feats)):
        feat = feats[i]
        div_lbls[labels[i]].append(feat)
    return div_lbls

def get_rf(X, y):
    clf = RandomForestClassifier(n_jobs=2)
    clf.fit(X, y)
    return clf
def get_adaboost():
    from sklearn.ensemble import AdaBoostClassifier
    clf = AdaBoostClassifier(n_estimators=50)
    return clf

def graph(X):
    pass    

def plot_elbow_method(X):
    start = time.time()
    k_range = range(1,1000,50) 
    #k_range = [1, 5, 20, 50, 100, 500, 1000]
    # Percentage of variance explained is the ratio of the between-group variance to the total variance, also known as an F-test.
    k_means_var = [get_kmeans_estimater(X, k) for k in k_range]
    centroids = [x.cluster_centers_ for x in k_means_var]
    k_euclid = [cdist(X, cent, 'euclidean') for cent in centroids]
    dist = [np.min(ke, axis=1) for ke in k_euclid]
    wcss = [sum(d**2) for d in dist]
    tss = sum(pdist(X)**2)/X.shape[0]
    bss = tss - wcss
    ratio = bss / tss * 100
    print ratio
    end = time.time()
    print 'plotting took: ', end - start
    plt.plot(k_range, ratio)
    plt.xlabel('Number of clusters')
    plt.ylabel('Percent of Variance explained')
    plt.show()

def find_label_mapping(labels, km_labels):
    from collections import Counter
    mapping = [[],[],[],[],[]]
    for i in range(len(labels)):
        mapping[km_labels[i]].append(labels[i])
    out = map(lambda x: Counter(x).most_common(1)[0][0], mapping)
    return out

def accuracy_test_km(X, y, noop_0, noop_1):
    k = 5
    est, y_hat = cluster_n(X, k)
    label_mapping = find_label_mapping(y, y_hat)
    y_hat = map(lambda x: label_mapping[x], y_hat)
    count = 0
    length = len(y)
    for i in range(length):
        #print(y[i], y_hat[i])
        if y[i] == y_hat[i]:
            count += 1
    return float(count) / float(length)

def do_performance_evaluation(X):
    for k in range(2, 20, 1):
        estimater, labels =  cluster_n(X, k)
        print(k, ' : ', silhouette_score(X, labels, metric='euclidean',sample_size=X.shape[0]))

def filter_words(text):
    words = text.split()
    filtered_words = [word for word in words if WORD_CHECKER.check(word)]
    return " ".join(filtered_words)

def accuracy_test_rf(X_tr, y_tr, X_t, y_t):
    rf_cl = get_rf(X_tr, y_tr)
    y_hat = rf_cl.predict(X_t) 
    length = len(y_hat)
    count = 0
    for i in range(0, len(y_hat)):
        if y_hat[i] == y_t[i]:
            count += 1
    return float(count) / float(length)
def cross_validation(X, y, n_fold, acc_test):
    accuracy = []
    tot = len(y)
    for i in range(1, n_fold):
        cur = tot / n_fold * i
        X_tr = X[:cur, :]
        y_tr = y[:cur]
        X_t = X[cur::, :]
        y_t = y[cur::]
        accuracy.append(acc_test(X_tr, y_tr, X_t, y_t))
    print sum(accuracy) / len(accuracy)
    return accuracy

def main():
    if not os.path.exists('X_file'):
        X_file = open('X_file', 'w')
        f_vals_file = open('f_vals_file', 'w')
        f_names_file = open('f_names_file', 'w')
        id_file = open('id_file', 'w')
        true_labels_file = open('true_labels_file', 'w')

        start = time.time()
        # Get data
        ids, articles, true_labels = get_data(2000) # specify how much you want
        end = time.time()
        print true_labels
        np.save(true_labels_file, true_labels)
        np.save(id_file, ids)
        articles = map(filter_words, articles)
        print 'parsing data took: ', end - start

        if testing:
            start = time.time()

        f_names_t, f_vals_t = featurize(articles)

        if testing:
            end = time.time()
            print 'featurizing data took: ', end - start
            print 'shape of raw data: ', f_vals_t.shape
            start = time.time()

        f_names, f_vals = filter_features(f_names_t, f_vals_t, THRESHHOLD)
        np.save(f_names_file, f_names)
        np.save(f_vals_file, f_vals)

        if testing:
            end = time.time()
            print 'filtering feature took: ', end - start
            start = time.time()

        X = tfidf_weights(f_vals)
        np.save(X_file, X)

        if testing:
            end = time.time()
            print 'tfidf weighting took: ', end - start
            print 'shape of featurized data: ', X.shape
    else:
        X_file = open('X_file', 'r')
        f_vals_file = open('f_vals_file', 'r')
        f_names_file = open('f_names_file', 'r')
        id_file = open('id_file', 'r')
        true_labels_file = open('true_labels_file', 'r')

        X = np.load(X_file)
        f_vals = np.load(f_vals_file)
        f_names = np.load(f_names_file)
        ids = np.load(id_file)
        true_labels = np.load(true_labels_file)
    y = true_labels

    # Random Forest Cross_validation
    cross_validation(X, y, 20, accuracy_test_rf)

    # Adaboost
    from sklearn.cross_validation import cross_val_score
    ada_cl = get_adaboost()
    scores = cross_val_score(ada_cl, X, y, cv=20)
    print scores.mean()

    # K Means Cross Validation
    cross_validation(X, true_labels, 20, accuracy_test_km)

    # Choose K using "Elbow method"/ F-Test 
    if testing:
        plot_elbow_method(X)
    
    # Number of clusters
    k = 5 # Choose k using elbow method

    # DO CLUSTERING HERE (LEGACY CODE)
    estimater, km_labels =  cluster_n(X, k)
    classification = dict(zip(ids, km_labels)) 
    div_lbls = get_output(km_labels, X, k) 

    # Performance Evaluation
    pca = PCA(n_components=2).fit(X.T)
    #do_performance_evaluation(pca.components_.T)
 
    # GRAPHING
    num_feats = len(f_names)
    reduced_X = PCA(n_components=2).fit_transform(X) 
    est_graph, lbls_graph = cluster_n(reduced_X, k)
    div_lbls_graph = get_output(lbls_graph, reduced_X, k)
    # Get 10 diff colors
    x = np.arange(k)
    ys = [i+x+(i*x)**2 for i in range(k)]
    colors = cm.rainbow(np.linspace(0, 1, len(ys)))

    for i in range(len(div_lbls)):
        cur = np.array(div_lbls_graph[i])
        plt.scatter(cur[:,0], cur[:,1], c=colors[i])
    centroids = est_graph.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='black', zorder=10)
    plt.show()

if __name__ == "__main__":
    main()
