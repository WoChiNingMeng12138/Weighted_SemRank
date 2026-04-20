import argparse
from transformers import AutoTokenizer
from classifier_utils import ClassModel, create_infer_dataset
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torch
import os
import itertools
from tqdm import tqdm
import json
import pickle

from corpus_io import load_corpus, resolve_data_dir

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='main', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model_path', type=str, help='model ckpt', default='classifier/topic_classifier_specter2.pt')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--model', type=str, default='allenai/specter2_base')
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
        help='Directory for specter2_topics.json and later artifacts. Default: ./LitSearch, ./CSFCube, or ./DORISMAE by dataset.',
    )
    parser.add_argument(
        '--corpus_jsonl',
        type=str,
        default=None,
        help='Optional path to the corpus file (CSFCube: .jsonl; DORISMAE: .jsonl or official pickle `corpus`).',
    )
    args = parser.parse_args()
    args.data_dir = resolve_data_dir(args.dataset, args.data_dir)
    os.makedirs(args.data_dir, exist_ok=True)

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device} (dataset={args.dataset}, data_dir={args.data_dir})')

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    state_dict = torch.load(args.model_path, map_location=device)
    num_label, emb_dim = state_dict['label_embedding_weights'].size()
    model = ClassModel(args.model, emb_dim, torch.empty((num_label, emb_dim))).to(device)
    model.load_state_dict(state_dict)
    model = model.to(device)

    id2doc, corpus, id2corpus_id = load_corpus(
        args.dataset, args.data_dir, corpus_jsonl=args.corpus_jsonl
    )

    test_data = create_infer_dataset(corpus, tokenizer)
    dataset = TensorDataset(test_data["input_ids"], test_data["attention_masks"])
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    with torch.no_grad():
        predictions = []
        for batch in tqdm(data_loader):
            input_ids = batch[0].to(device)
            input_mask = batch[1].to(device)
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
        doc = id2doc[corpus_id]
        results[corpus_id] = {
            'corpus_id': corpus_id,
            'title': doc.get('title', ''),
            'abstract': doc.get('abstract', ''),
            'topic_labels': [(id2label[i], id2name[i], float(pred[i])) for i in topic_label_rank[:100]]
        }

    json.dump(results, open(f'{args.data_dir}/specter2_topics.json', 'w'))
#     pickle.dump(results, open(f'{args.data_dir}/specter2_topics.pkl', 'wb'))