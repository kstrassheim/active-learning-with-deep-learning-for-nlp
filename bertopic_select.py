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

from bertopic import BERTopic
topic_model = BERTopic(embedding_model="all-mpnet-base-v2")
topic_model.fit(X_train)

def get_bert_topic_args(sentences, samples:int):
    l = len(sentences)
    if (l <= samples): return list(range(0, l))
    selected = []
    raw_topics = topic_model.transform(sentences)
    topics = {}
    for i in range(len(raw_topics[0])):
        t = raw_topics[0][i]
        if t not in topics:
            topics[t] = []
        topics[t].append(i)

    # use round robin to get diversity terms from every topic
    while len(selected) < samples:
        for t in topics:
            if len(topics[t]) > 0:
                pick = topics[t][0]
                if pick not in selected:
                    selected.append(pick)
                    topics[t].remove(pick)
                    if len(selected) >= samples:
                        return selected
    return selected

def apply_active_learning(algorithm, source, source_y, sample_size=100, epochs=3, continuous_mode=False, verbose=True):
    res = []
    source = list(source)
    source_y = list(source_y)
    i = 0
    samples = []
    samples_y = []
    model = None
    while len(source) > 0:
        if verbose : print(f'Active Learning {"Continuous" if continuous_mode else ""} round: {i}')
        start_round = timer()
        if continuous_mode:
            samples = []
            samples_y = []
        start_al = timer()
        pick_args = algorithm(source, sample_size)

        #sort reverse or pop will end with argument out of range exception
        pick_args.sort(reverse=True)
        
        duration_al = timedelta(seconds=timer()-start_al)
        if verbose : print(f'Active Learning {"Continuous" if continuous_mode else ""} AL duration: {duration_al}')
        # transfer samples from embedding list to samples
        for d in pick_args: 
            samples.append(source.pop(d))
            samples_y.append(source_y.pop(d))
            
        start_ml = timer()   
        metric = {} #, trainer, model = train_model(samples, samples_y, X_val, y_val, X_test, y_test, epochs=epochs, model=model if continuous_mode else None)
        duration_ml = timedelta(seconds=timer()-start_ml)
        if verbose : print(f'Active Learning {"Continuous" if continuous_mode else ""} ML duration: {duration_ml}')
        if continuous_mode:
            i = i + len(samples)
            metric["trained_samples"] = i
        else:
            i = i + len(samples)
            metric["trained_samples"] = len(samples)
        
        duration_total = timedelta(seconds=timer()-start_round)
        metric["duration_al"] = duration_al
        metric["duration_ml"] = duration_ml
        metric["duration_total"] = duration_total
        res.append(metric)
    return pd.DataFrame(res)
start_t = timer()
met = apply_active_learning(get_bert_topic_args, X_train, y_train, continuous_mode=True)
print(timedelta(seconds=timer()-start_t))
# res_bt = get_bert_topic_args(X_train, samples=1000)
# print(len(res_bt))
# print(res_bt)