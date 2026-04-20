import os
import torch

from tqdm import tqdm, trange
from transformers import AutoTokenizer
import pickle
import numpy as np
import json

import argparse

from adapters import AutoAdapterModel

from corpus_io import load_corpus, resolve_data_dir, specter2_encode_text_for_doc


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='main', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--dataset',
        type=str,
        default='litsearch',
        choices=('litsearch', 'csfcube', 'dorismae'),
        help='Corpus: LitSearch from HuggingFace, or CSFCube / DORISMAE from local corpus.jsonl (see README).',
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default=None,
        help='Directory for embeddings and phrase files. Default: ./LitSearch, ./CSFCube, or ./DORISMAE by dataset.',
    )
    parser.add_argument(
        '--corpus_jsonl',
        type=str,
        default=None,
        help='Optional path to the corpus file (CSFCube: .jsonl; DORISMAE: .jsonl or official pickle `corpus`).',
    )
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--model', type=str, default='allenai/specter2_base')
    args = parser.parse_args()
    args.data_dir = resolve_data_dir(args.dataset, args.data_dir)
    os.makedirs(args.data_dir, exist_ok=True)

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device} (dataset={args.dataset}, data_dir={args.data_dir})')

    id2doc, _, id2corpus_id = load_corpus(
        args.dataset, args.data_dir, corpus_jsonl=args.corpus_jsonl
    )
    # Paper Table 2 "SPECTER-v2" uses the proximity adapter, not raw specter2_base (see allenai/specter2 README).
    corpus = [specter2_encode_text_for_doc(id2doc[cid]) for cid in id2corpus_id]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoAdapterModel.from_pretrained(args.model)
    model.load_adapter("allenai/specter2", source="hf", load_as="proximity")
    model.set_active_adapters("proximity")
    # Adapter weights load on CPU; move the full model after adapters are attached.
    model.to(device)

    # Get corpus embeddings
    corpus_embeddings = []
    batch_size = 16
    with torch.no_grad():
        for start_idx in trange(0, len(corpus), batch_size):
            texts = corpus[start_idx:start_idx+batch_size]
            inputs = tokenizer(
                texts,
                padding=True,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                return_token_type_ids=False,
            ).to(device)
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
            inputs = tokenizer(
                texts,
                padding=True,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                return_token_type_ids=False,
            ).to(device)
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
            inputs = tokenizer(
                texts,
                padding=True,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                return_token_type_ids=False,
            ).to(device)
            model_out = model(**inputs)
            embeddings = model_out.last_hidden_state[:, 0, :]
            phrase_embeddings.append(embeddings.cpu())
    phrase_embeddings = torch.cat(phrase_embeddings, 0)
    print(phrase_embeddings.size())
    torch.save(phrase_embeddings, f'{args.data_dir}/specter2_corpus_with-topic-terms.json.phrase-enc-specter.pt')