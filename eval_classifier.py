import argparse
from transformers import AutoTokenizer
from classifier_utils import ClassModel, create_infer_dataset
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torch
import os
import itertools
from tqdm import tqdm
import datasets
import json
import pickle

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='main', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model_path', type=str, help='model ckpt', default='classifier/topic_classifier_specter2.pt')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--model', type=str, default='allenai/specter2_base')
    parser.add_argument('--data_dir', type=str, default='./LitSearch')
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    state_dict = torch.load(args.model_path)
    num_label, emb_dim = state_dict['label_embedding_weights'].size()
    model = ClassModel(args.model, emb_dim, torch.empty((num_label, emb_dim))).to(f'cuda:{args.gpu}')
    model.load_state_dict(state_dict)
    model = model.to(f'cuda:{args.gpu}')

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

#     # Load from local corpus
#     corpus_data = []
#     with open(f'{args.data_dir}/corpus.jsonl') as f:
#         for line in f:
#             corpus_data.append(json.loads(line.strip()))
#     id2doc = {doc['corpus_id']: doc for doc in corpus_data}
#     corpus = []
#     id2corpus_id = []
#     for paper in corpus_data:
#         i = paper['corpus_id']
#         id2corpus_id.append(i)
#         corpus.append(paper['text'])

    test_data = create_infer_dataset(corpus, tokenizer)
    dataset = TensorDataset(test_data["input_ids"], test_data["attention_masks"])
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    with torch.no_grad():
        predictions = []
        for batch in tqdm(data_loader):
            input_ids = batch[0].to(f'cuda:{args.gpu}')
            input_mask = batch[1].to(f'cuda:{args.gpu}')
            output = model(input_ids, input_mask).cpu().numpy()
            predictions.append(output)
    predictions = np.concatenate(predictions, axis=0)


    id2label, label2id, id2name = {}, {}, {}
    with open('classifier/labels.txt') as f:
        for i, line in enumerate(f):
            label, name, _ = line.strip().split('\t')
            id2label[i] = label
            label2id[label] = i
            id2name[i] = name

    results = {}
    for i, pred in enumerate(predictions):
        corpus_id = id2corpus_id[i]
        topic_label_rank = np.argsort(-pred)
        results[corpus_id] = {
            'corpus_id': corpus_id,
            'title': id2doc[corpus_id]['title'],
            'abstract': id2doc[corpus_id]['abstract'],
            'topic_labels': [(id2label[i], id2name[i], float(pred[i])) for i in topic_label_rank[:100]]
        }

    json.dump(results, open(f'{args.data_dir}/specter2_topics.json', 'w'))
#     pickle.dump(results, open(f'{args.data_dir}/specter2_topics.pkl', 'wb'))