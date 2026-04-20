import json
import os
import numpy as np
import argparse

from corpus_io import resolve_data_dir


def build_specter2_corpus_with_topic_terms(data_dir: str, output: dict) -> dict:
    """
    Map LLM <top>...</top> names onto classifier topic tuples.

    Keys in topic_labels are canonical names (mixed case). LLM output is matched
    case-insensitively on the label string (strip + lower).
    """
    new_results: dict = {}
    for corpusid, info in output.items():
        if info.get('llm_error') or not info.get('llm_output'):
            continue
        topic_by_lower = {}
        for t in info['topic_labels']:
            k = t[1].strip().lower()
            if k not in topic_by_lower:
                topic_by_lower[k] = t
        llm_output = info['llm_output']
        raw = llm_output.split('<top>')[1].split('</top>')[0].split(', ')
        topics = []
        for tname in raw:
            key = tname.strip().lower()
            if key in topic_by_lower:
                topics.append(topic_by_lower[key])
        terms = llm_output.split('<kp>')[1].split('</kp>')[0].split(', ')
        terms = [t.strip() for t in terms]
        new_results[corpusid] = {
            'corpusid': corpusid,
            'title': info.get('title', ''),
            'abstract': info.get('abstract', ''),
            'text': f"{info['title']}. {info['abstract']}",
            'topics': topics,
            'terms': terms,
        }
    out_path = os.path.join(data_dir, 'specter2_corpus_with-topic-terms.json')
    json.dump(new_results, open(out_path, 'w'), indent=0)
    return new_results


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='main', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--dataset',
        type=str,
        default='litsearch',
        choices=('litsearch', 'csfcube', 'dorismae'),
        help='Must match the dataset used for eval_classifier / specter2_topics.json (default dirs: see README).',
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default=None,
        help='Directory with specter2_topics.json. Default: ./LitSearch, ./CSFCube, or ./DORISMAE by dataset.',
    )
    parser.add_argument(
        '--provider',
        type=str,
        choices=('openai', 'tamu'),
        default=os.environ.get('CHAT_API_PROVIDER', 'openai'),
        help='LLM backend: openai (OPENAI_API_KEY) or tamu (TAMUS_AI_CHAT_API_KEY / TAMU_CHAT_API_KEY).',
    )
    parser.add_argument(
        '--gpt_model',
        type=str,
        default='gpt-4.1-mini',
        help='Model id: OpenAI e.g. gpt-4.1-mini; TAMU default gpt-4.1-mini → protected.gpt-4.1-mini (override id via TAMU_GPT_4_1_MINI_MODEL or pass e.g. --gpt_model protected.gpt-4.1-mini).',
    )
    parser.add_argument('--tier', type=str, default='tier1')
    parser.add_argument(
        '--half_usage',
        action='store_true',
        help='Halve client-side RPM/TPM caps (recommended when TAMU or OpenAI still returns quota/throttle errors).',
    )
    parser.add_argument(
        '--rebuild_final_only',
        action='store_true',
        help='Only rebuild specter2_corpus_with-topic-terms.json from existing specter2-llm-topics.json (no API).',
    )
    args = parser.parse_args()
    args.data_dir = resolve_data_dir(args.dataset, args.data_dir)
    os.makedirs(args.data_dir, exist_ok=True)

    if args.rebuild_final_only:
        llm_path = os.path.join(args.data_dir, 'specter2-llm-topics.json')
        output = json.load(open(llm_path))
        n = len(build_specter2_corpus_with_topic_terms(args.data_dir, output))
        print(f'Wrote specter2_corpus_with-topic-terms.json ({n} papers) under {args.data_dir}')
        raise SystemExit(0)

    from api.openai.chat import chat, CHAT_FAILED_RESPONSE

    topic_candidates = json.load(open(f'{args.data_dir}/specter2_topics.json'))

    instruction = '''
    You will receive a paper abstract along with a set of candidate topics for the paper. 
    Your first task is to select the topics that best align with the core theme of the paper.
    Exclude topics that are too broad or less relevant. 
    Only use the topic names in the candidate set. 
    Your second task is to generate a complete list of key phrases extracted from the paper.
    Do some rationalization before outputting the list of relevant topics and key phrases.
    Output format: '<top> topic 1, topic 2, ... </top><kp>key phrase 1, key phrase 2, ... </kp>'.
    '''

    try:
        output = json.load(open(f'{args.data_dir}/specter2-llm-topics.json'))
    except:
        output = {}

    id_list = [i for i in topic_candidates if i not in output]
    curr_len = len(id_list)
    batch_size = 5000
    while len(id_list) > 0:
        print(f'{len(output)}/{len(topic_candidates)}')
        inputs = []
        for corpus_id in id_list[:batch_size]:
            paper = topic_candidates[corpus_id]
            title = paper['title']
            abstract = paper['abstract']
            topic_labels = [l[1] for l in paper['topic_labels']]
            topic_labels = ', '.join(topic_labels[:100])
            doc = f'Title: {title}\nPaper Abstract: {abstract}\nCandidate Topics: {topic_labels}'
            inputs.append(doc)

        results = chat(
            inputs,
            instruction,
            model_name=args.gpt_model,
            tier_list=args.tier,
            api_provider=args.provider,
            half_usage=args.half_usage,
            seed=np.random.randint(0, 1000),
        )

        for corpus_id, res in zip(id_list[:batch_size], results):
            if res == CHAT_FAILED_RESPONSE:
                output[corpus_id] = {**topic_candidates[corpus_id], 'llm_error': 'api_error', 'llm_output': None}
                continue
            if '<kp>' not in res or '<top>' not in res or\
            '</kp>' not in res or '</top>' not in res:
                continue
            output[corpus_id] = topic_candidates[corpus_id]
            output[corpus_id]['llm_output'] = res

        print('start saving')
        json.dump(output, open(f'{args.data_dir}/specter2-llm-topics.json', 'w'), indent=0)
        print('finished saving')

        id_list = [i for i in topic_candidates if i not in output]

        if len(id_list) == curr_len:
            break
        else:
            curr_len = len(id_list)

    build_specter2_corpus_with_topic_terms(args.data_dir, output)