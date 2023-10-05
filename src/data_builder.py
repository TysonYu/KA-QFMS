from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import torch
import torch.nn.functional as F
import jsonlines
from nltk.tokenize import sent_tokenize
import pickle
import os

class OurDataset(Dataset):
    """Summarization dataset"""
    def __init__(self, args, mode):
        self.args = args
        self.mode = mode
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model)
        original_data_path = './data/qmsum/processed_data_no_sent_split' + '/' + mode + '.jsonl'
        self.src, self.tgt = self.file_reader(original_data_path)
        if os.path.exists(self.args.data_path + '/' + self.mode + '_12_dual_rank.pkl'):
            with open(self.args.data_path + '/' + self.mode + '_12_dual_rank.pkl', 'rb') as f:
                self.segmented_src = pickle.load(f)
        else:
            print('don not find error processed data file')
            exit(0)

        if self.args.knowledge_aware != '':
            if os.path.exists(self.args.data_path + '/' + self.mode + '_12_dual_rank_keyphrases.pkl'):
                with open(self.args.data_path + '/' + self.mode + '_12_dual_rank_keyphrases.pkl', 'rb') as f:
                    self.triples = pickle.load(f)
            else:
                print('don not find error. triple data file')
                exit(0)

        print('Segmentation done. Number of semigments: ')
        print([len(item) for item in self.segmented_src])

    def __len__(self):
        return len(self.segmented_src)

    def __getitem__(self, idx):
        if self.args.knowledge_aware != '':
            return self.segmented_src[idx], self.tgt[idx], self.triples[idx]
        else:
            return self.segmented_src[idx], self.tgt[idx]

    def file_reader(self, file_path):
        with jsonlines.open(file_path, 'r') as reader:
            src = []
            tgt = []
            for obj in reader:
                src.append(obj['src'])
                tgt.append(obj['tgt'])
        return src, tgt

    def get_query(self, input):
        query = []
        src = []
        for item in input:
            temp = item.split('</s>')
            query.append(temp[0].replace('<s>','').strip())
            src.append(temp[1].replace('<s>','').strip())
        return query, src

    def collate_fn(self, data):
        # rebuild the raw text and truncate to max length
        raw_src = [pair[0][:6] for pair in data]
        raw_tgt = [pair[1][:6] for pair in data]
        # get querys
        querys = []
        for item in raw_src:
            querys.append(item[0].split('</s>')[0])

        # get docs without querys
        docs_without_query = []
        for item in raw_src:
            this_doc = []
            for para in item:
                this_doc.append(para.split('</s>')[-1])
            docs_without_query += this_doc

        batch_size = len(raw_tgt)
        new_raw_src = []
        for item in raw_src:
            new_raw_src += item
        raw_src = new_raw_src
        raw_ids = self.tokenizer(raw_src, add_special_tokens=True, padding='max_length', truncation=True, max_length=self.args.max_input_len, return_tensors='pt')

        src_ids = raw_ids['input_ids']
        src_mask = raw_ids['attention_mask']

        raw_tgt_ids = self.tokenizer(raw_tgt, add_special_tokens=True, padding=True, truncation=True, max_length=self.args.max_output_len, return_tensors='pt')['input_ids']
        if 'bart' in self.args.model:
            from transformers.models.bart.modeling_bart import shift_tokens_right
            tgt_ids = shift_tokens_right(raw_tgt_ids, 1, 2)
            raw_tgt_ids[raw_tgt_ids[:, :] == 1] = -100
        if 'pegasus' in self.args.model:
            from transformers.models.pegasus.modeling_pegasus import shift_tokens_right
            tgt_ids = shift_tokens_right(raw_tgt_ids, 0, 0)
        label_ids = raw_tgt_ids

        # Knowledge-aware part
        if self.args.knowledge_aware != '':
            doc_with_query_triple = []
            raw_triples = [pair[2] for pair in data][0]
            for i in range(len(raw_triples)):
                this_query = querys[0]
                this_keywords = raw_triples[i]
                this_keywords = this_keywords.split('</s>')
                this_keywords = list(set(this_keywords))
                this_keywords = [item for item in this_keywords[:10] if len(item) > 3]
                this_keywords = " ".join(this_keywords)
                this_doc = docs_without_query[i]
                this_doc = this_query + "</s></s>" + this_keywords + "</s></s>" + this_doc
                doc_with_query_triple.append(this_doc)
            knowledge_inputs = self.tokenizer(doc_with_query_triple, add_special_tokens=True, padding='max_length', truncation=True, max_length=self.args.max_input_len, return_tensors='pt')


            querys = self.tokenizer(querys, add_special_tokens=True, padding=True, truncation=True, max_length=256, return_tensors='pt')
            docs_without_query = self.tokenizer(docs_without_query, add_special_tokens=True, padding='max_length', truncation=True, max_length=self.args.max_input_len, return_tensors='pt')

        else:
            knowledge_inputs = None
            querys = None
            docs_without_query = None

        return {'src_ids': [src_ids, knowledge_inputs, querys, docs_without_query],
                'mask': src_mask,
                'decoder_ids': tgt_ids,
                'label_ids': label_ids,
                'labels':raw_tgt,}