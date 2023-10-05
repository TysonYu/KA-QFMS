import pickle
from nltk.corpus import stopwords
out_words = {}
for word in stopwords.words():
    out_words[word] = 1
from tqdm import tqdm

number_of_segment = 8
mode = 'train'

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

with open('processed_data_turn_split/{}_all_seg_with_triple.pkl'.format(mode), 'rb') as f:
    data = pickle.load(f)
with open('processed_data_turn_split/{}_rankscores.pkl'.format(mode), 'rb') as f:
    qa_scores = pickle.load(f)

new_data = []
new_seg_input = []
new_triples = []
for i in tqdm(range(len(data))):
    keys = list(data[i].keys())
    a = qa_scores[i][0]
    this_qascores = [(float(i)-min(a))/(max(a)-min(a)) for i in a]
    ka_scores = []
    for key in keys:
        ka_scores.append(data[i][key][-1])
    a = ka_scores
    if max(a)-min(a) == 0:
        pass
    else:    
        ka_scores = [(float(i)-min(a))/(max(a)-min(a)) for i in a]
    all_scores = []
    for j in range(len(ka_scores)):
        all_scores.append(ka_scores[j]+this_qascores[j])
    for j in range(len(keys)):
        this_seg_info = {}
        this_seg_info['triples'] = data[i][keys[j]]
        this_seg_info['qa_score'] = this_qascores[j]
        this_seg_info['ka_score'] = ka_scores[j]
        this_seg_info['all_scores'] = all_scores[j]
        # key_phrase = triple2text(data[i][keys[j]][0])
        # this_seg_info['key_phrase'] = key_phrase
        data[i][keys[j]] = this_seg_info
    
    new = {k: v for k, v in sorted(data[i].items(), key=lambda item: item[1]['all_scores'], reverse=True)}
    new_data.append(new)
    new_keys = list(new.keys())
    if len(new) >= number_of_segment:
        new_seg_input.append(new_keys[:number_of_segment])
        new_triples.append([triple2text(new[new_keys[i]]['triples'][0]) for i in range(number_of_segment)])
    else:
        new_seg_input.append(new_keys)
        new_triples.append([triple2text(new[new_keys[i]]['triples'][0]) for i in range(len(new_keys))])

with open('processed_data_turn_split/{}_all_seg_with_triple_with_scores.pkl'.format(mode), 'wb') as f:
    pickle.dump(new_data, f)

with open('processed_data_turn_split/{}_8_dual_rank.pkl'.format(mode), 'wb') as f:
    pickle.dump(new_seg_input, f)

with open('processed_data_turn_split/{}_8_dual_rank_keyphrases.pkl'.format(mode), 'wb') as f:
    pickle.dump(new_triples, f)
