import config
import json
import pickle
import random

from transformers import AutoTokenizer as Tokenizer

from util import _find_sub_list

# BERT
tokenizer = Tokenizer.from_pretrained(config.bert_dir, do_lower_case=False)

# Roberta
# tokenizer = Tokenizer.from_pretrained("deepset/roberta-base-squad2")

def to_bert_example(query, document, start, end, max_seq_length=512):

    query = tokenizer.tokenize('[CLS] ' + query + ' [SEP]')
    q_sub_tokens = tokenizer.convert_tokens_to_ids(query)
    len_q_sub_tokens = len(q_sub_tokens)


    d_sub_tokens = []
    map_to_origin = []
    for idx, word in enumerate(document):
        sub_tokens = tokenizer.tokenize(word)
        sub_tokens = tokenizer.convert_tokens_to_ids(sub_tokens)
        d_sub_tokens.extend(sub_tokens)
        map_to_origin.extend([idx] * len(sub_tokens))

    start_sub_idx = map_to_origin.index(start)
    end_sub_idx = len(d_sub_tokens) if end + 1 not in map_to_origin else map_to_origin.index(end + 1)
    end_sub_idx = end_sub_idx - 1 # inclusive

    start_sub_idx += len_q_sub_tokens
    end_sub_idx += len_q_sub_tokens

    while not end_sub_idx < max_seq_length: # to ensure the answer is in the context
        t = random.randint(1, 100) 
        start_sub_idx -= t
        end_sub_idx -= t
        d_sub_tokens = d_sub_tokens[t:]
        map_to_origin = map_to_origin[t:]

        # print('Here')
    
    input_ids = q_sub_tokens + d_sub_tokens

    if len(input_ids) > max_seq_length:
        input_ids = input_ids[:max_seq_length]

    segment_ids = [0] * len_q_sub_tokens + [1] * (len(input_ids) - len_q_sub_tokens)
    input_mask = [1] * len(input_ids)
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    input_mask += padding
    segment_ids += padding
    # label_id = label2id[example.label]
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    return [input_ids, input_mask, segment_ids, start_sub_idx, end_sub_idx], map_to_origin, len_q_sub_tokens



# def parse_elem(elem):
#     # dict_keys(['scene_id', 'utterances', 'plots', 'rc_entities', 'span_qa']) rc_entities
#     # to do, print(elem['rc_entities'])
#     # print(elem['scene_id'])
#     utterances = elem['utterances']

#     sp_set = {}
#     speaker_set = {}
#     pos_set = {}

#     all_tokens = []
#     all_qa_pairs = []

#     for idx, utterance in enumerate(utterances):
#         # to do, utterance['character_entities']
#         speakers = utterance['speakers']

#         if len(speakers) > 0:
#             speakers = ' '.join(speakers) + ' :'
#             speakers = speakers.split(' ')

#         pos_set[(idx, -1)] = len(all_tokens)
#         all_tokens.extend(speakers) 
#         pos_set[(idx, -2)] = len(all_tokens) - 2

#         if len(utterance['speakers']) == 1:
#             if not utterance['speakers'][0] in sp_set:
#                 sp_set[utterance['speakers'][0]] = 'a'
#                 speaker_set[len(speaker_set) + 1] = (pos_set[(idx, -1)], pos_set[(idx, -2)])

#         if len(utterance['tokens']) > 0:
#             token = utterance['tokens']
#         else:
#             token = utterance['tokens_with_note']
        
#         token = [item for sublist in token for item in sublist]
#         for i, t in enumerate(token):
#             pos_set[(idx, i)] = len(all_tokens)
#             all_tokens.append(t)
    

#     for qa in elem['span_qa']:
#         qa_id = qa['id']
#         question = qa['question']
#         answers = qa['answers']
        
#         answers_poses = []

#         for ans in answers:
#             answer_text = ans['answer_text']
#             utterance_id = ans['utterance_id']
#             inner_start = ans['inner_start']
#             inner_end = ans['inner_end']
#             is_speaker = ans['is_speaker']

#             if is_speaker:
#                 inner_start, inner_end = -1, -2

#             answers_poses.append((pos_set[(utterance_id, inner_start)], pos_set[(utterance_id, inner_end)], answer_text))

#         all_qa_pairs.append((question, qa_id, answers_poses))

#     return all_tokens, all_qa_pairs, speaker_set


def read_data_enhance(filename):
    with open(filename) as filein:
        data = json.loads(filein.read())
        results = []
        for d in data:
            try:
                text, relation, knowledge = d
            except:
                print(d)
                continue
            all_tokens, all_qa_pairs, speaker_set = parse_elem(knowledge)

            # speaker_set will be used further

            bert_examples = []
            for elem in all_qa_pairs:
                question, qa_id, answers = elem
                for ans in answers:
                    res, map_to_origin = to_bert_example(question, all_tokens, ans[0], ans[1])
                    bert_examples.append(res)
            results.append([all_tokens, all_qa_pairs, bert_examples])
        return results


def _align_re_with_qa():
    aligned_examples = {}
    for filename in ['train.json', 'dev.json', 'test.json']:
        with open('data/enhanced_dialogre_data/' + filename) as filein:
            data = json.loads(filein.read())
            results = []
            for d in data:
                try:
                    text, relation, knowledge = d
                except:
                    print(d)
                    continue
                s_id = knowledge['scene_id']
                aligned_examples.setdefault(s_id, list())
                aligned_examples[s_id].extend(relation)
    return aligned_examples


def parse_elem_v2(elem):
    utterances = elem['paragraphs']

    utrs = utterances[0]['utterances:']
    qas = utterances[0]['qas']

    sp_set = {}
    speaker_set = {}
    pos_set = {}

    all_tokens = []
    all_qa_pairs = []

    for idx, utterance in enumerate(utrs):
        speakers = utterance['speakers']

        if len(speakers) > 0:
            speakers = ' '.join(speakers) + ' :'
            speakers = speakers.split(' ')

        pos_set[(idx, -1)] = len(all_tokens)
        all_tokens.extend(speakers) 
        pos_set[(idx, -2)] = len(all_tokens) - 2  # ':' and inclusive

        if len(utterance['speakers']) == 1:
            if not utterance['speakers'][0] in sp_set:
                sp_set[utterance['speakers'][0]] = 'a'
                speaker_set[len(speaker_set) + 1] = (pos_set[(idx, -1)], pos_set[(idx, -2)])
        
        token = utterance['utterance'].split()
        for i, t in enumerate(token):
            pos_set[(idx, i)] = len(all_tokens)
            all_tokens.append(t)
    
    for qa in qas:
        qa_id = qa['id']
        question = qa['question']
        answers = qa['answers']
        
        answers_poses = []

        for ans in answers:
            answer_text = ans['answer_text']
            utterance_id = ans['utterance_id']
            inner_start = ans['inner_start']
            inner_end = ans['inner_end']
            is_speaker = ans['is_speaker']

            if is_speaker:
                inner_start, inner_end = -1, -2

            # print(answer_text)
            # print(' '.join(all_tokens[pos_set[(utterance_id, inner_start)]: pos_set[(utterance_id, inner_end)]+1]))
            try:
                answers_poses.append((pos_set[(utterance_id, inner_start)], pos_set[(utterance_id, inner_end)], answer_text))
            except:
                # adv examples may throw error...
                pass

        all_qa_pairs.append((question, qa_id, answers_poses))

    return all_tokens, all_qa_pairs, speaker_set


def _to_idx(x, speaker_set, all_tokens):
    if x.startswith('Spea'):
        mapped_id = int(x.split()[1])
        if mapped_id in speaker_set:
            return speaker_set[mapped_id]
        return 0, 0
    else:
        s, e = _find_sub_list(x.split(), all_tokens)
        return (s, e) if s != -1 else (0, 0)


def read_data_friends(filename, raw_relations={}):
    with open(filename) as filein:
        data = json.loads(filein.read())
        results = []
        for d in data['data']:
            
            all_tokens, all_qa_pairs, speaker_set = parse_elem_v2(d)

            s_id = d['title']
            relation_list = raw_relations.get(s_id, [])
                        
            relation_result = []
            for r in relation_list:
                x, y, rid = r['x'], r['y'], r['rid'][0]
                if rid == 37:
                    continue
                x_s, x_e = _to_idx(x, speaker_set, all_tokens)
                y_s, y_e = _to_idx(y, speaker_set, all_tokens)

                relation_result.append([x_s, x_e, y_s, y_e, rid])
            print(len(relation_result))

            bert_examples = []
            for elem in all_qa_pairs:
                question, qa_id, answers = elem
                for ans in answers:
                    bert_feature, map_to_origin, len_q_sub_tokens = to_bert_example(question, all_tokens, ans[0], ans[1])
                    
                    r_res = []
                    for relation in relation_result:
                        try:
                            x_s, x_e, y_s, y_e, r_id = relation
                            x_s = map_to_origin.index(x_s)
                            y_s = map_to_origin.index(y_s)
                            if x_s >=0 and y_s >=0 and x_s < max_seq_length and y_s < max_seq_length:
                                r_res.append([x_s + len_q_sub_tokens, y_s + len_q_sub_tokens, r_id])
                        except:
                            pass

                    bert_examples.append([question, qa_id, ans, bert_feature, map_to_origin, len_q_sub_tokens, r_res])
            results.append([all_tokens, bert_examples])
        return results


def align_coreference():
    res = {}
    for f in [  'data/character-identification/json/character-identification-trn.json', 
                'data/character-identification/json/character-identification-dev.json', 
                'data/character-identification/json/character-identification-tst.json']:
        with open(f) as filein:
            data = json.loads(filein.read())
            for d in data:
                for elem in data['episodes']:
                    key = elem['episode_id']
                    scenes = elem['scenes']
                    for scene in scenes:
                        utterance = scene['utterances']
                        for utter in utterance:
                            c = utter['character_entities']
                            res.setdefault(key, 0)
                            for mention in c:
                                if len(mention) != 0:
                                    res[key] += 1

                            # for mention in c:
                            #     res.setdefault(key, 0)
                            #     res[key] += len(mention)

    return res

        

if __name__ == '__main__':
    #### version 0
    # train_examples = read_data_enhance('data/qa_train.json')
    # dev_examples = read_data_enhance('data/qa_dev.json')
    # test_examples = read_data_enhance('data/qa_test.json')

    # all_examples = [train_examples, dev_examples, test_examples]
    # with open('data/data.pickle', 'wb') as f:
    #     pickle.dump(all_examples, f, pickle.HIGHEST_PROTOCOL)


    #### version 2
    aligned_examples = _align_re_with_qa()
    res = {}
    for elem in aligned_examples:
        key = elem.split('_')[1]
        res.setdefault(key, 0)
        res[key] += len(aligned_examples[elem])

    num_train, num_dev, num_test = 0, 0, 0
    for elem in res:
        print(elem, res[elem])
        if int(elem[1:]) <= 20:
            num_train += res[elem]
        elif 21 <= int(elem[1:]) <=22:
            num_dev += res[elem]
        else:
            num_test += res[elem]
    
    print(num_train, num_dev, num_test)

    res = align_coreference()
    num_train, num_dev, num_test = 0, 0, 0
    for elem in res:
        print(elem, res[elem])
        key = int(elem.split('_')[1][1:])
        if key <= 20:
            num_train += res[elem]
        elif 21 <= key <=22:
            num_dev += res[elem]
        else:
            num_test += res[elem]
    print(num_train, num_dev, num_test)


    # train_examples = []
    # # train_examples = read_data_friends('data/FriendsQA/dat/friendsqa_trn.json', aligned_examples)
    # dev_examples = read_data_friends('data/FriendsQA/dat/friendsqa_dev_adv.json', aligned_examples)
    # test_examples = read_data_friends('data/FriendsQA/dat/friendsqa_tst_adv.json', aligned_examples)

    # all_examples = [train_examples, dev_examples, test_examples]
    # with open('data/data_original.pickle', 'wb') as f:
    #     pickle.dump(all_examples, f, pickle.HIGHEST_PROTOCOL)


    