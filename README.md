# Hybrid Losses for Hierarchical Embedding Learning

Code for the ICASSP 2025 paper titled "Hybrid Losses for Hierarchical Embedding Learning"

## Abstract
In traditional supervised learning, the cross-entropy loss treats all incorrect predictions equally, ignoring the relevance or proximity of wrong labels to the correct answer. By leveraging a tree hierarchy for fine-grained labels, we investigate hybrid losses, such as generalised triplet and cross-entropy losses, to enforce similarity between labels within a multi-task learning framework. We propose metrics to evaluate the embedding space structure and assess the model’s ability to generalise to unseen classes, that is, to infer similar classes for data belonging to unseen categories. Our experiments on OrchideaSOL, a four- level hierarchical instrument sound dataset with nearly 200 detailed categories, demonstrate that the proposed hybrid losses outperform previous works in classification, retrieval, embedding space structure, and generalisation.

## Setup
To install dependencies:
```
pip install -r requirements.txt
```

## Data
Download the OrchideaSOL dataset from [here](https://forum.ircam.fr/projects/detail/orchideasol/)

## Train
Train models with a config file:
```
python scripts/main.py --config configs/fold0/pl.yaml
```
