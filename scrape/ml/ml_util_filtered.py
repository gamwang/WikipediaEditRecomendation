from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import KMeans
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
import matplotlib.cm as cm

THRESH_HOLD = 7
WORD_CHECKER = enchant.Dict("en_US")

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
    estimater = KMeans(init='k-means++', n_clusters=n, n_init=10)
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

def graph(X):
    pass

def filter_words(text):
    words = text.split()
    filtered_words = [word for word in words if WORD_CHECKER.check(word)]
    return " ".join(filtered_words)

def plot_elbow_method(X):
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
    # Get data
    articles, ids = get_data(1000) # specify how much you want
    #data_mapping = ["pageid0", "pageid1", "pageid2", "pageid3", "pageid4"]
    #data = ["James likes red peach. James likes red peach.", "Jon likes that James likes red peach. Jon likes that James likes red peach.", "James likes red peach.", "James likes red peach. Haha", "James likes red peach. No way"]
    filtered_articles = map(filter_words, articles)
    # f_names_t, f_vals_t = featurize(articles)
    f_names_t, f_vals_t = featurize(filtered_articles)
    print f_vals_t.shape
    f_names, f_vals = filter_features(f_names_t, f_vals_t, THRESH_HOLD)
    print f_vals
    X = tfidf_weights(f_vals)

    # Choose K using "Elbow method"/ F-Test
    #
    #plot_elbow_method(X)

    # Number of clusters
    k = 9

    # DO CLUSTERING HERE (LEGACY CODE)
    estimater, labels =  cluster_n(X, k)
    end = time.time()
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

    """
    num_feats_sqrt = math.ceil(num_feats ** (0.5))
    for j in range(1, num_feats):
        colors = ['r', 'b', 'g', 'y']
        cur_placement = num_feats_sqrt * 100 + num_feats_sqrt * 10 + j
        print cur_placement
        plt.subplot(cur_placement)
        for i in range(len(div_lbls)):
            cur = np.array(div_lbls[i])
            plt.scatter(cur[:,j-1], cur[:,j], c=colors[i])
        print f_names[j-1], f_names[j]
        plt.xlabel('weighted frequencies of ' + f_names[j-1])
        plt.ylabel('weighted frequencies of ' + f_names[j])
    plt.show()
    """

if __name__ == "__main__":
    main()
