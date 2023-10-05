from openie import StanfordOpenIE
from nltk.corpus import stopwords
from tqdm import tqdm_gui

from tqdm import tqdm
import torch
import torch.nn.functional as F
import os
import copy
import json
import pickle

properties = {
    'openie.affinity_probability_cap': 2 / 3,
}

with open( 'processed_data_turn_split/test_all_seg.pkl', 'rb') as f:
    segmented_input = pickle.load(f)

with StanfordOpenIE(properties=properties) as client:
    new_segmented_input = []
    for number, one_segmented_input in tqdm(enumerate(segmented_input)):
        query_without_stopwords = " ".join([item for item in one_segmented_input[0].split('</s></s>')[0].split() if item not in stopwords.words()])
        print(query_without_stopwords)
        print(len(one_segmented_input))
        this_doc_dic = {}
        for i,item in enumerate(segmented_input[number]):
            origin_item = copy.deepcopy(item)
            item = item.split('</s></s>')[-1]
            counter = 0
            triples = []
            for triple in client.annotate(item):
                for word in triple['subject'].split():
                    if word in query_without_stopwords and word not in stopwords.words():
                        counter += 1
                        print(triple, query_without_stopwords)
                        triples.append(triple)
                for word in triple['object'].split():
                    if word in query_without_stopwords and word not in stopwords.words():
                        counter += 1
                        triples.append(triple)
                        print(triple, query_without_stopwords)
            this_doc_dic[origin_item] = [triples, counter]
        new_segmented_input.append(this_doc_dic)


with open('256/train_all_seg_with_triple.pkl', 'wb') as f:
    pickle.dump(new_segmented_input, f)