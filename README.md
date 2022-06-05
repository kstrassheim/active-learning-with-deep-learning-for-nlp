# Active Learning with Deep Learning for NLP (2022)
We present our concept of a new type of Active-Learning for Deep Learning with NLP text classification and experimentally prove its performance against Random Sampling as well as its runtime performance on the Security Threat dataset from CySecAlert. These new Active Learning algorithms are based on Sentence-BERT and BERTopic clustering algorithms with allow us to generate fixed length tokens for whole sentences to make them comparable to each other. Further the Tokens are Clustered using K-Means or HDBScan to get diverse clusters to pick the samples out of them.

# Purpose
This project shows the possibility and validates the performance of both new diversity based clustering algorithms compared to random-sampling in an experiment.

## Abstract-Keywords
Active-Learning, Deep-Learning, Sentence-BERT, BERTopic, NLP

## Technical Keywords
Python-3, Jupyter-Notebooks, Pandas, PyTorch, HuggingFace-Lib

## Manual (Short)
The 2 Active-Learning algorithms and the experimental runs are defined in experiments.py. The runtime is really long (About 6 days with RTX 3090). Afterward the experiments have to be converted to fit into the evaluation format using convert-result.py.

The final results can be regarded in result-analysis.ipynb