# precalculate encoded sentences to achieve a better performance on sbert
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
random_seed = 1337
from timeit import default_timer as timer
from datetime import timedelta

#get and prepare training data
def get_training_data(dataset_path:str, test_split_ratio:float=0.1,verbose=False):
    data = pd.read_json(dataset_path)
    data["label_train"] = data["label"] - 1
    data["display_text"] = [d[1]['text'][d[1]['displayTextRangeStart']: d[1]['getDisplayTextRangeEnd']] for d in data[["text","displayTextRangeStart", "getDisplayTextRangeEnd"]].iterrows()]
    if verbose : print("max text length", len(data.iloc[np.argmax(data['text'].to_numpy())]['text']))
    max_display_text_length = len(data.iloc[np.argmax(data['display_text'].to_numpy())]['display_text'])
    if verbose : print("max display text length", max_display_text_length)
    X = data.display_text.to_list()
    y = data.label_train.to_list()
    # split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split_ratio, random_state=random_seed, shuffle=True)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=test_split_ratio * len(X) / len(X_train), random_state=random_seed, shuffle=True)
    return X_train, y_train, X_val, y_val, X_test, y_test

X_train, y_train, X_val, y_val, X_test, y_test = get_training_data('data/dataset_1.json', test_split_ratio=0.2)

def sbert_tokenize(sentences, verbose=False, bert_model_name='all-mpnet-base-v2'):
    from sentence_transformers import SentenceTransformer, util
    import torch
    model = SentenceTransformer(bert_model_name)
    model.max_seq_length = np.argmax(sentences)
    
    embedding_list = model.encode(sentences, show_progress_bar=verbose)
    return embedding_list

encoded_sentences = sbert_tokenize(X_train)
encoded_sentences_dict = {X_train[i]:encoded_sentences[i] for i in range(len(encoded_sentences))}

def get_sbert_centroid_args(sentences, num_labels:int, bert_model_name='all-mpnet-base-v2', verbose=False):
    l = len(sentences)
    if l <= 0: return []
    
    # if sample size is smaller than the list there is nothing to sample  then return all indices
    if l < num_labels: return list(range(0, l))
    
    # encode embeddings
    embedding_list = []
    try:
        embedding_list = [encoded_sentences_dict[s] for s in sentences]
    except:
        # if list cannot be encoded with precalculated list - load new SentenceTransformer
        from sentence_transformers import SentenceTransformer, util
        import torch
        model = SentenceTransformer(bert_model_name)
        model.max_seq_length = np.argmax(sentences)
        embedding_list = model.encode(sentences, show_progress_bar=verbose)
    
    from sklearn.cluster import KMeans
    
    clustering_model = KMeans(n_clusters=num_labels, random_state=1337) 
    clustering_model.fit(embedding_list)
    cluster_assignment = clustering_model.labels_
    clustered_sentences = {}
    for sentence_id, cluster_id in enumerate(cluster_assignment):
        if cluster_id not in clustered_sentences:
            clustered_sentences[cluster_id] = []

        clustered_sentences[cluster_id].append(sentence_id)

    centroids = []
    for i in range(len(clustering_model.cluster_centers_)):
        center = clustering_model.cluster_centers_[i]
        # get centroid arg for cluster by min euclidian distance from cluster center
        centroid_arg = clustered_sentences[i][np.argmin([np.linalg.norm(embedding_list[cluster_item_arg]-center) for cluster_item_arg in clustered_sentences[i]])]
        centroids.append(centroid_arg)
    return centroids

centroid_args = get_sbert_centroid_args(sentences=X_train, num_labels=5)
centroid_args.sort()
centroid_args