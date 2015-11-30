from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
import simplejson as json
from matplotlib import pyplot as plt
import numpy as np
import time
import operator
import math

THRESH_HOLD = 5

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

def get_output(labels, feats):
    feats = np.array(feats)
    div_lbls = [[],[],[],[]]
    for i in range(len(feats)):
        feat = feats[i]
        div_lbls[labels[i]].append(feat)
    return div_lbls

def main():
    # Get data
    articles, ids = get_data(1000) # specify how much you want
    #data_mapping = ["pageid0", "pageid1", "pageid2", "pageid3", "pageid4"]
    #data = ["James likes red peach. James likes red peach.", "Jon likes that James likes red peach. Jon likes that James likes red peach.", "James likes red peach.", "James likes red peach. Haha", "James likes red peach. No way"]
    start = time.time()
    f_names_t, f_vals_t = featurize(articles)
    f_names, f_vals = filter_features(f_names_t, f_vals_t, THRESH_HOLD)
    X = tfidf_weights(f_vals)
    """
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)
    distances, indices = nbrs.kneighbors(X)
    
    plt.hist(distances[:,1])
    plt.show()
    """
    db = DBSCAN(eps=1.0, min_samples=5, algorithm='ball_tree').fit(X)
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    print n_clusters_

    """
    print('Estimated number of clusters: %d' % n_clusters_)
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
    print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
    print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
    print("Adjusted Rand Index: %0.3f"
          % metrics.adjusted_rand_score(labels_true, labels))
    print("Adjusted Mutual Information: %0.3f"
          % metrics.adjusted_mutual_info_score(labels_true, labels))
    print("Silhouette Coefficient: %0.3f"
          % metrics.silhouette_score(X, labels))
    print n_clusters_
    """


if __name__ == "__main__":
    main()
