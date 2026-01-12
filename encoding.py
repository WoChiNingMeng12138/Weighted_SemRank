import torch

from tqdm import tqdm, trange
from transformers import AutoTokenizer, AutoModel
import pickle
import numpy as np
import datasets
import json

import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='main', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_dir', type=str, default='./LitSearch')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--model', type=str, default='allenai/specter2_base')
    args = parser.parse_args()

    corpus_data = datasets.load_dataset("princeton-nlp/LitSearch", "corpus_clean", split="full")
    id2doc = {doc['corpusid']: doc for doc in corpus_data}
    corpus = []
    id2corpus_id = []
    for paper in corpus_data:
        i = paper['corpusid']
        id2corpus_id.append(i)
        title = paper['title']
        abstract = paper['abstract']
        corpus.append(f'{title}. {abstract}')

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(f'cuda:{args.gpu}')

    # Get corpus embeddings
    corpus_embeddings = []
    batch_size = 16
    with torch.no_grad():
        for start_idx in trange(0, len(corpus), batch_size):
            texts = corpus[start_idx:start_idx+batch_size]
            inputs = tokenizer(texts, padding=True, return_tensors='pt', \
                                    truncation=True, max_length=512).to(f'cuda:{args.gpu}')
            model_out = model(**inputs)
            embeddings = model_out.last_hidden_state[:, 0, :]
            corpus_embeddings.append(embeddings.cpu())

    torch.save(corpus_embeddings, f'{args.data_dir}/corpus-enc-specter.pt')
    pickle.dump(id2corpus_id, open(f'{args.data_dir}/corpus-enc-index.pkl', 'wb'))


    # Load topic label space
    id2label, label2id, id2name, name2id = {}, {}, [], {}
    with open('classifier/labels.txt') as f:
        for i, line in enumerate(f):
            label, name, _ = line.strip().split('\t')
            id2label[i] = label
            label2id[label] = i
            id2name.append(name)
            name2id[name] = i

    # Topic embeddings
    class_embeddings = []
    batch_size = 32
    with torch.no_grad():
        for start_idx in trange(0, len(id2name), batch_size):
            texts = id2name[start_idx:start_idx+batch_size]
            inputs = tokenizer(texts, padding=True, return_tensors='pt', \
                                    truncation=True, max_length=512).to(f'cuda:{args.gpu}')
            model_out = model(**inputs)
            embeddings = model_out.last_hidden_state[:, 0, :]
            class_embeddings.append(embeddings.cpu())
    class_embeddings = torch.cat(class_embeddings, 0)
    torch.save(class_embeddings, f'{args.data_dir}/topic-enc-specter.pt')


    # Loaded corpus with labels
    corpus_with_labels = json.load(open(f'{args.data_dir}/specter2_corpus_with-topic-terms.json'))
    # get all phrase embedding
    id2phrase = []
    phrase2id = {}
    for corpusid in tqdm(corpus_with_labels):
        for term in corpus_with_labels[corpusid]['terms']:
            if term == '': continue
            term = term.lower()
            if term not in name2id and term not in phrase2id:
                phrase2id[term] = len(id2phrase)
                id2phrase.append(term)
    print(len(id2phrase))
    pickle.dump((id2phrase, phrase2id), open(f'{args.data_dir}/specter2_corpus_with-topic-terms.json.phrase_idx.pkl', 'wb'))

    phrase_embeddings = []
    batch_size = 32
    with torch.no_grad():
        for start_idx in trange(0, len(id2phrase), batch_size):
            texts = id2phrase[start_idx:start_idx+batch_size]
            inputs = tokenizer(texts, padding=True, return_tensors='pt', \
                                    truncation=True, max_length=512).to(f'cuda:{args.gpu}')
            model_out = model(**inputs)
            embeddings = model_out.last_hidden_state[:, 0, :]
            phrase_embeddings.append(embeddings.cpu())
    phrase_embeddings = torch.cat(phrase_embeddings, 0)
    print(phrase_embeddings.size())
    torch.save(phrase_embeddings, f'{args.data_dir}/specter2_corpus_with-topic-terms.json.phrase-enc-specter.pt')