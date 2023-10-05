from transformers import AutoTokenizer, AutoModel
import jsonlines
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
import torch
import torch.nn.functional as F
import os
import copy
import json
import pickle
import numpy as np
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
from transformers.models.bart.modeling_bart import shift_tokens_right

os.environ['CUDA_VISIBLE_DEVICES'] = '6'
def file_reader(file_path):
    with jsonlines.open(file_path, 'r') as reader:
        src = []
        tgt = []
        for obj in reader:
            src.append(obj['src'])
            tgt.append(obj['tgt'])
    return src, tgt

def get_query(input):
    query = []
    src = []
    for item in input:
        temp = item.split('</s>')
        query.append(temp[0].replace('<s>','').strip())
        src.append(temp[1].replace('<s>','').strip())
    return query, src

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
#Encode text
def encode(texts):
    # Tokenize sentences
    encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
    encoded_input['input_ids'] = encoded_input['input_ids'].cuda()
    # encoded_input['token_type_ids'] = encoded_input['token_type_ids'].cuda()
    encoded_input['attention_mask'] = encoded_input['attention_mask'].cuda()
    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input, return_dict=True)
    # Perform pooling
    embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    # Normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings


tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/multi-qa-mpnet-base-cos-v1")
model = AutoModel.from_pretrained("sentence-transformers/multi-qa-mpnet-base-cos-v1").cuda()


from nltk.corpus import stopwords
out_words = {}
for word in stopwords.words():
    out_words[word] = 1


def triple2text(triples):
    words = []
    for triple in triples:
        for item in triple['subject'].split():
            if item not in words:
                if item.isalpha():
                    if item not in stopwords.words():
                        words.append(item)
    for triple in triples:
        for item in triple['relation'].split():
            if item not in words:
                if item.isalpha():
                    if item not in stopwords.words():
                        words.append(item)
    for triple in triples:
        for item in triple['object'].split():
            if item not in words:
                if item.isalpha():
                    if item not in stopwords.words():
                        words.append(item)
    output = "</s>".join(words)
        # this_triple_text = triple['subject'] + '</s>' + triple['relation'] +  '</s>' + triple['object'] + '</s>'
        # output += this_triple_text
    return output
    
with open('processed_data_turn_split/test_all_seg_with_triple.pkl', 'rb') as f:
    data = pickle.load(f)
print('finish reading data')
new_seg_input = []
triples = []
number_of_segment = 16
all_scores = []

for doc in tqdm(data):
    this_scores = []
    keys = list(doc.keys())
    query = keys[0].split('</s></s>')[0]
    this_doc = [item.split('</s></s>')[-1] for item in keys]
    
    #Encode query and docs
    query_emb = encode(query)
    doc_emb = encode(this_doc)
    #Compute dot score between query and all document embeddings
    scores = torch.mm(query_emb, doc_emb.transpose(0, 1))[0].cpu().tolist()
    knowledge_score = []
    for key in keys:
        knowledge_score.append(doc[key][1])
    knowledge_score = knowledge_score / np.linalg.norm(knowledge_score)
    merge_score = []
    for i in range(len(knowledge_score)):
        merge_score.append(knowledge_score[i] + scores[i])
    
    for i in range(len(keys)):
        doc[keys[i]].append(merge_score[i])

    new = {k: v for k, v in sorted(doc.items(), key=lambda item: item[1][-1], reverse=True)}
    new_seg_list = list(new.keys())
    this_scores.append(scores)
    this_scores.append(knowledge_score)
    all_scores.append(this_scores)
    
    if len(new_seg_list) < number_of_segment:
        new_seg_input.append(new_seg_list + ['' for i in range(number_of_segment - len(new_seg_list))])
        triples.append([triple2text(new[new_seg_list[i]][0]) for i in range(len(new_seg_list))] + ['' for i in range(number_of_segment - len(new_seg_list))])
    else:
        new_seg_input.append(new_seg_list[:number_of_segment])
        triples.append([triple2text(new[new_seg_list[i]][0]) for i in range(number_of_segment)])

with open('512/test_16_single_rank.pkl', 'wb') as f:
    pickle.dump(new_seg_input, f)

with open('512/test_16_single_rank_keyphrases.pkl', 'wb') as f:
    pickle.dump(triples, f)

# with open('processed_data_turn_split/val_rankscores.pkl', 'wb') as f:
#     pickle.dump(all_scores, f)