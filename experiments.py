import pandas as pd
import numpy as np
import math
from sklearn.model_selection import  StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
# libs for time measurement
from timeit import default_timer as timer
from datetime import timedelta

def get_cysec_dataset(dataset_path:str, random_state=0, delete_balance_rate=None):
    data = pd.read_json(dataset_path)
    if delete_balance_rate is not None:
        todelete = math.ceil(len(data[data['label']==1]) * delete_balance_rate)
        print(todelete)
        #  clean some labels to get balance
        import random
        random.seed(random_state) 
        data = data.drop(data[data['label']==1].iloc[random.sample(range((len(data[data['label']==1]))), todelete)].index)
    
    data["label_train"] = data["label"] - 1
    data["label_bin"] = data['label_train'].apply(lambda x: 1 if x > 0 else 0)
    data["label_tri"] = data['label_train'].apply(lambda x: 2 if x > 2 else x)
    data["display_text"] = [d[1]['text'][d[1]['displayTextRangeStart']: d[1]['getDisplayTextRangeEnd']] for d in data[["text","displayTextRangeStart", "getDisplayTextRangeEnd"]].iterrows()]

    return data

#get and prepare training data
def split_training_data(X, y, n_splits, random_state:int, test_split_ratio:float, verbose=False):

    X = np.array(X)
    y = np.array(y)
    # print(y.tolist())
    # split data
    
    X_train = []
    y_train = []
    X_test = []
    y_test = []

    for train_index, test_index in StratifiedShuffleSplit(n_splits=n_splits, test_size=test_split_ratio, random_state=random_state).split(X, y):
        X_train.append(X[train_index].tolist())
        y_train.append(y[train_index].tolist())
        X_test.append(X[test_index].tolist())
        y_test.append(y[test_index].tolist())

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split_ratio, random_state=random_seed, shuffle=True)
    # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=test_split_ratio * len(X) / len(X_train), random_state=random_seed, shuffle=True)
    if n_splits == 1:
        return X_train[0], y_train[0], X_test[0], y_test[0]
    else:
        return X_train, y_train, X_test, y_test

# #get and prepare training data
# def get_training_data(data, n_splits, X_train_col='display_text', y_label_col="label_train", test_split_ratio:float=0.1, verbose=False):

#     X = data[X_train_col].to_numpy()
#     y = data[y_label_col].to_numpy()
#     # print(y.tolist())
#     # split data
    
#     X_train = []
#     y_train = []
#     X_test = []
#     y_test = []

#     for train_index, test_index in StratifiedShuffleSplit(n_splits=n_splits, test_size=test_split_ratio, random_state=random_seed).split(X, y):
#         X_train.append(X[train_index].tolist())
#         y_train.append(y[train_index].tolist())
#         X_test.append(X[test_index].tolist())
#         y_test.append(y[test_index].tolist())

#     # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split_ratio, random_state=random_seed, shuffle=True)
#     # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=test_split_ratio * len(X) / len(X_train), random_state=random_seed, shuffle=True)
#     return X_train, y_train, X_test, y_test, data

def compute_metrics(pred, labels):
    pred = np.argmax(pred, axis=1)
    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred, average='weighted')
    precision = precision_score(y_true=labels, y_pred=pred, average='weighted')
    f1 = f1_score(y_true=labels, y_pred=pred, average='weighted')
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

def train_model(X_train, y_train, X_val, y_val, X_test, y_test, random_state, batch_size=128, epochs=3, model=None, tokenizer=None):

    # BEGIN disable logging 
    import logging
    def set_global_logging_level(level=logging.ERROR, prefices=[""]):
        import re
        prefix_re = re.compile(fr'^(?:{ "|".join(prefices) })')
        for name in logging.root.manager.loggerDict:
            if re.match(prefix_re, name):
                logging.getLogger(name).setLevel(level)
    set_global_logging_level(logging.CRITICAL) # disable INFO and DEBUG logging everywhere
    
    import warnings
    warnings.filterwarnings("ignore")
    # END disable logging
    
    # BEGIN Set determinism !! must be inside function in every loop to work

    from os import environ
    environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    # !! important !! import torch after setting cublas deterministic or it will not work !!
    import torch
    from transformers import TrainingArguments, Trainer, DistilBertTokenizer, DistilBertForSequenceClassification
    import transformers
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    import random
    random.seed(random_state)
    
    # END Set determinism
    
     # Create torch dataset
    class Dataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels=None):
            self.encodings = encodings
            self.labels = labels
        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            if self.labels: item["labels"] = torch.tensor(self.labels[idx])
            return item
        def __len__(self):
            return len(self.encodings["input_ids"])
    
    #disable logging
    #transformers.logging.set_verbosity(transformers.logging.CRITICAL)
    
    # create tokenizer
    if tokenizer is None:
        tokenizer = DistilBertTokenizer.from_pretrained('./distilbert-base-uncased') 
    
    # create datasets
    train_dataset = Dataset(tokenizer(X_train, truncation=True, padding=True, max_length=512), y_train)
    val_dataset = Dataset(tokenizer(X_val, truncation=True, padding=True, max_length=512), y_val)
    test_dataset = Dataset(tokenizer(X_test, padding=True, truncation=True, max_length=512), y_test)
    
    #create model

    if model is None:
        model = DistilBertForSequenceClassification.from_pretrained('./distilbert-base-uncased', num_labels=len(np.unique(y_train)))


    #training settings
    args = TrainingArguments(
        output_dir="output",
        evaluation_strategy="epoch",
        eval_steps=1,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        seed=random_state,
        load_best_model_at_end=False
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=lambda p: compute_metrics(p[0], p[1])
    )
    # disable print log
    from transformers.trainer_callback import PrinterCallback
    trainer.remove_callback(PrinterCallback)

    # Train
    trainer.train()

    # Test
    metrics = trainer.evaluate(test_dataset, metric_key_prefix="")

#     raw_pred, _, _ = trainer.predict(test_dataset)
#     m = compute_metrics(raw_pred, y_test)
    return metrics, trainer, trainer.model, tokenizer

class bertopic_clustering:

    def __init__(self, sample_size:int, X):
        self.__sample_size = sample_size
        self.result = None

        from bertopic import BERTopic
        self.__topic_model = BERTopic(embedding_model="all-mpnet-base-v2")
        self.__topic_model.fit(X)

    def run(self, input, random_state, verbose=False):
        self.result = None
        l = len(input)
        if (l <= self.__sample_size): return list(range(0, l))
        selected = []
        raw_topics = self.__topic_model.transform(input)
        topics = {}
        for i in range(len(raw_topics[0])):
            t = raw_topics[0][i]
            if t not in topics:
                topics[t] = []
            topics[t].append(i)

        # use round robin to get diversity terms from every topic
        while len(selected) < self.__sample_size:
            for t in topics:
                if len(topics[t]) > 0:
                    pick = topics[t][0]
                    if pick not in selected:
                        selected.append(pick)
                        topics[t].remove(pick)
                        if len(selected) >= self.__sample_size:
                            self.result = selected
                            return
        self.result = selected

class sbert_kmeans:
    def __init__(self, sample_size:int, X, bert_model_name='all-mpnet-base-v2'):
        self.__sample_size = sample_size
        self.__bert_model_name = bert_model_name
        self.result = None

        # presave encodings for faster processing
        # set determinism
        from os import environ
        environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        import torch
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False

        # cache results
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(bert_model_name)
        model.max_seq_length = np.argmax(X)
        encoded_sentences = model.encode(X, show_progress_bar=False)
        self._encoded_sentences_dict = {X[i]:encoded_sentences[i] for i in range(len(encoded_sentences))}
    
    def run(self, input, random_state, verbose=False):
        l = len(input)
        if l <= 0: return []
        
        # if sample size is smaller than the list there is nothing to sample  then return all indices
        if l < self.__sample_size: return list(range(0, l))
        
        # encode embeddings
        embedding_list = []
        try:
            if self._encoded_sentences_dict is not None:
                embedding_list = [self._encoded_sentences_dict[s] for s in input]
        except:
            # if list cannot be encoded with precalculated list - load new SentenceTransformer
            from sentence_transformers import SentenceTransformer
            import torch
            model = SentenceTransformer(self.__bert_model_name)
            model.max_seq_length = np.argmax(input)
            embedding_list = model.encode(input, show_progress_bar=verbose)
        
        from sklearn.cluster import KMeans
        
        clustering_model = KMeans(n_clusters=self.__sample_size, random_state=random_state) 
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
        self.result = centroids

class random_sampling:
    def __init__(self, sample_size:int,):
        self.__sample_size = sample_size
        self.result = None

    def run(self, input, random_state, verbose=False):
        l = len(input)
        if l <= 0: return []
        # if sample size is smaller than the list there is nothing to sample then return all indices
        if l < self.__sample_size: return list(range(0, l))
        import random
        random.seed(random_state) 
        self.result = random.sample(range(0, l), self.__sample_size)


def apply_active_learning(algorithm, source, source_y, sample_size, random_state, ml_x_val, ml_y_val, ml_x_test, ml_y_test, stop_at=None, ml_batch_size=100, ml_epochs=3, continuous_mode=True, title="AL", verbose=False):
    res = []
    source = list(source)
    source_y = list(source_y)
    i = 1
    samples = []
    samples_y = []
    model = None
    tokenizer = None
    if stop_at is None: stop_at = len(source)
    while len(source) > 0 and i * sample_size <= stop_at:
        trained_samples = sample_size * i
        #if verbose : print(f'AL {title} {"Continuous" if continuous_mode else ""} processing_samples: {trained_samples}')
        start_round = timer()
        if continuous_mode:
            samples = []
            samples_y = []
        start_al = timer()
        algorithm.run(source, random_state, verbose)
        pick_args = algorithm.result
        #sort reverse or pop will end with argument out of range exception
        pick_args.sort(reverse=True)
        
        duration_al = timedelta(seconds=timer()-start_al)
        #if verbose : print(f'AL {title} {"Continuous" if continuous_mode else ""} - AL duration: {duration_al}')
        # transfer samples from embedding list to samples
        for d in pick_args: 
            samples.append(source.pop(d))
            samples_y.append(source_y.pop(d))
            
        start_ml = timer()   
        metric, trainer, model, tokenizer = train_model(X_train = samples, y_train= samples_y,  X_val= ml_x_val, y_val= ml_y_val, X_test= ml_x_test, y_test= ml_y_test, random_state=random_state,  batch_size=ml_batch_size, epochs=ml_epochs, model=model if continuous_mode else None, tokenizer=tokenizer)
        duration_ml = timedelta(seconds=timer()-start_ml)
        #if verbose : print(f'AL {title} {"Continuous" if continuous_mode else ""} - ML duration: {duration_ml}')
        metric["trained_samples"] = trained_samples
        
        duration_total = timedelta(seconds=timer()-start_round)
        if verbose: print(f'{title}{"-C" if continuous_mode else ""} - Samples:{trained_samples} - Duration: {duration_total} AL:{duration_al} ML:{duration_ml}', end="\r")
        metric["duration_al"] = duration_al
        metric["duration_ml"] = duration_ml
        metric["duration_total"] = duration_total
        res.append(metric)
        i = i + 1
    return pd.DataFrame(res)

def run_experiments(data, train_col, label_col, sample_size, n_splits = 2, stop_at=None, ml_batch_size=100, ml_epochs=3, random_states=[0], verbose=True):
    start_ex = timer()
    res_rand =[] 
    res_sbert = []
    res_bt = []
    for ri, r in enumerate(random_states):
        X_train_list, y_train_list, X_tests_list, y_tests_list = split_training_data(data[train_col], data[label_col], n_splits=n_splits, test_split_ratio=0.4, random_state=r) 
        bt = bertopic_clustering(sample_size, X = data[train_col].to_list())
        sk = sbert_kmeans(sample_size=sample_size, X = data[train_col].to_list())
        rd = random_sampling(sample_size=sample_size)
        res_rand_f = []
        res_sbert_f = []
        res_bt_f = []
    
        for i, X_train in enumerate(X_train_list):
            X_test, y_test, X_val, y_val = split_training_data( X_tests_list[i], y_tests_list[i], n_splits=1, test_split_ratio=0.5, random_state=r) 
            try:
                res_rand_f.append(apply_active_learning(algorithm=rd, source=X_train, source_y=y_train_list[i], ml_x_val=X_val, ml_y_val=y_val, ml_x_test=X_test, ml_y_test=y_test, random_state=r, stop_at=stop_at, sample_size=sample_size, ml_batch_size=ml_batch_size, ml_epochs=ml_epochs, title=f'RD-{i}-{ri}', verbose=verbose)) 
            except Exception as e:
                print(ri, i, 'RD', e)
            try:
                res_sbert_f.append(apply_active_learning(algorithm=sk, source=X_train, source_y=y_train_list[i], ml_x_val=X_val, ml_y_val=y_val, ml_x_test=X_test, ml_y_test=y_test, random_state=r, stop_at=stop_at, sample_size=sample_size, ml_batch_size=ml_batch_size, ml_epochs=ml_epochs, title=f'SK-{i}-{ri}', verbose=verbose))
            except Exception as e:
                print(ri, i, 'SK', e)
            try:
                res_bt_f.append(apply_active_learning(algorithm=bt, source=X_train, source_y=y_train_list[i], ml_x_val=X_val, ml_y_val=y_val, ml_x_test=X_test, ml_y_test=y_test, random_state=r, stop_at=stop_at, sample_size=sample_size, ml_batch_size=ml_batch_size, ml_epochs=ml_epochs, title=f'BT-{i}-{ri}', verbose=verbose))
            except Exception as e:
                print(ri, i, 'BT', e)
        res_rand.append(res_rand_f)
        res_sbert.append(res_sbert_f)
        res_bt.append(res_bt_f)
    # print_plot(res_rand, res_sbert, res_bt, title=f'Sample Size {sample_size}')
    duration = timedelta(seconds=timer()-start_ex)
    return {
            'res_rand': res_rand, 
            'res_sbert': res_sbert, 
            'res_bt': res_bt, 
            'duration' : duration
        }

if __name__ == "__main__":

    # SET Determinism
        # set determinism
    from os import environ
    environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False

    # BEGIN disable logging 
    import logging
    def set_global_logging_level(level=logging.ERROR, prefices=[""]):
        import re
        prefix_re = re.compile(fr'^(?:{ "|".join(prefices) })')
        for name in logging.root.manager.loggerDict:
            if re.match(prefix_re, name):
                logging.getLogger(name).setLevel(level)
    set_global_logging_level(logging.CRITICAL) # disable INFO and DEBUG logging everywhere
    
    import warnings
    warnings.filterwarnings("ignore")
    # END disable logging

    data = get_cysec_dataset('data/dataset_1.json', random_state=1337)
    ex = run_experiments(data, train_col='display_text', label_col='label_train', sample_size=50, ml_batch_size=25, ml_epochs=3, stop_at=100, random_states=[0, 42])
    print(ex)