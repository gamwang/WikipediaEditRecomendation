from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, MiniBatchKMeans
import simplejson as json
from matplotlib import pyplot as plt
import numpy as np
import operator
import math
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, cdist
import matplotlib.cm as cm
from sets import Set

def get_data(count):
    f = open('./articles.json', 'r')
    objs = f.read().split('\n')
    ids = []
    articles = []
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
        ids.append(pid)
        articles.append(extract)
        i += 1
    return articles, ids

def get_kmeans_estimater(data, n):
    estimater = KMeans(init='k-means++', n_clusters=n, n_init=10)
    estimater.fit(data)
    return estimater

def tfidf_featurize(data):
    featurizer = TfidfVectorizer(analyzer='word', ngram_range=(1,3), stop_words='english', min_df=0.01)
    X = featurizer.fit_transform(data)
    f_names = featurizer.get_feature_names()
    return f_names, X

def filter_features(f_names, f_vals):
    keep_list = Set()
    for i in range(len(f_vals)):
        for j in range(len(f_vals[i])):
            cur = f_vals[i][j]
            if j not in keep_list:
                keep_list.add(j)
    out_names = []
    out_vals = []
    keep_list = list(keep_list)
    for i in range(len(f_vals)):
        out_vals.append(operator.itemgetter(keep_list)(f_vals[i]))
    for i in range(len(f_names)):
        if i in keep_list:
            out_names.append(f_names[i])
    return out_names, out_vals

def cluster_n(data, n):
    estimater = KMeans(init='k-means++', n_clusters=n, n_init=10)
    estimater.fit(data)
    labels = estimater.labels_
    return estimater, labels

def get_output(labels, feats, k):
    feats = np.array(feats)
    div_lbls = []
    for i in range(k):
        div_lbls.append([])
    for i in range(len(feats)):
        feat = feats[i]
        div_lbls[labels[i]].append(feat)
    return div_lbls

def plot_elbow_method(X):
    X = X.toarray()
    k_range = range(1,30,3)
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
    plt.plot(k_range, ratio)
    plt.xlabel('Number of clusters')
    plt.ylabel('Percent of Variance explained')
    plt.show()

def main():
    articles, ids = get_data(1000)
    f_names, X = tfidf_featurize(articles)

    # Choose K using "Elbow method"/ F-Test
    #
    #plot_elbow_method(X)

    # Number of clusters
    k = 9

    # DO CLUSTERING HERE (LEGACY CODE)
    X = X.toarray()
    estimater, labels =  cluster_n(X, k)
    classification = dict(zip(ids, labels))
    div_lbls = get_output(labels, X, k)

    # GRAPHING
    num_feats = len(f_names)
    reduced_X = PCA(n_components=2).fit_transform(X)
    est_graph, lbls_graph = cluster_n(reduced_X, k)
    div_lbls_graph = get_output(lbls_graph, reduced_X, k)
    # Get 10 diff colors
    x = np.arange(10)
    ys = [i+x+(i*x)**2 for i in range(10)]
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
