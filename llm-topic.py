import json
import os
from api.openai.chat import chat
import numpy as np
import argparse

os.environ["OPENAI_API_KEY"] = 'your key'


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='main', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_dir', type=str, default='./LitSearch')
    parser.add_argument('--gpt_model', type=str, default='gpt-4.1-mini')
    parser.add_argument('--tier', type=str, default='tier3')
    args = parser.parse_args()

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

        results = chat(inputs, instruction, model_name=args.gpt_model, tier_list=args.tier, seed=np.random.randint(0, 1000))

        for corpus_id, res in zip(id_list[:batch_size], results):
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

    parsed_results = {}
    for corpusid, info in output.items():
        topic2score = {t[1]:t for t in info['topic_labels']}
        llm_output = info['llm_output']
        topics = llm_output.split('<top>')[1].split('</top>')[0].split(', ')
        topics = [topic2score[tname.strip().lower()] for tname in topics if tname.strip().lower() in topic2score]
        terms = llm_output.split('<kp>')[1].split('</kp>')[0].split(', ')
        terms = [t.strip() for t in terms]
        new_results[corpusid] = {
            'corpusid': corpusid,
            'text': info['text'],
            'topics': topics,
            'terms': terms
        }
    json.dump(new_results, open(f'{args.data_dir}/specter2_corpus_with-topic-terms.json', 'w'), indent=0)