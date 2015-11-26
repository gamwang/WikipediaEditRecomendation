from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import KMeans
import simplejson as json
from matplotlib import pyplot
import numpy as np
import time
import operator

THRESH_HOLD = 10

def featurize(data):
    ngram_vectorizer = CountVectorizer(analyzer='word', ngram_range=(2, 2), min_df=1)
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
    f_names_t, f_vals_t = featurize(data)
    f_names, f_vals = filter_features(f_names_t, f_vals_t, THRESH_HOLD)
    tfidf_vals = tfidf_weights(f_vals)
    est = get_kmeans_estimater(tfidf_vals, n)
    labels = est.labels_
    return est, labels, tfidf_vals, f_names

def main():
    f = open('./articles.json', 'r')
    objs = f.read().split('\n')
    ids = []
    articles = []
    i = 0
    for obj in objs:
        if i > 1000:
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
    #data_mapping = ["pageid0", "pageid1", "pageid2", "pageid3", "pageid4"]
    #data = ["James likes red peach. James likes red peach.", "Jon likes that James likes red peach. Jon likes that James likes red peach.", "James likes red peach.", "James likes red peach. Haha", "James likes red peach. No way"]
    start = time.time()
    estimater, labels, feats, feat_names = cluster_n(articles, 4)
    end = time.time()
    print end - start
    classification = dict(zip(ids, labels))
    feats = np.array(feats)
    div_lbls = [[],[],[],[]]
    for i in range(len(feats)):
        feat = feats[i]
        div_lbls[labels[i]].append(feat)
    colors = ['r', 'b', 'g', 'y']
    print feat_names
    # need a way to visualize dimension of data better; currently only visualizing feature #0 and 1
    for i in range(len(div_lbls)):
        cur = np.array(div_lbls[i])
        pyplot.scatter(cur[:,0], cur[:,1], c=colors[i])
    print(cur.shape)
    pyplot.show()
    #print classification

if __name__ == "__main__":
    main()
