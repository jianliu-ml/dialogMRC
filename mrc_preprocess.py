import config
import pickle
from util import _find_sub_list_all
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained(config.bert_dir, do_lower_case=False)

entity_transfer_set = {}

for i in range(20):
    entity_transfer_set['Speaker %d' % i] = '[unused%d]' % i


relation_questions = [
    # ('gpe:residents_of_place', 'who resides at #'),
    # ('gpe:visitors_of_place', 'who visits #'),
    # ('org:employees_or_members', 'who works at #'),
    # ('org:students', 'who studies at #'),
    ('per:acquaintance', 'Who is the acquaintance of # ?'),
    ('per:age', 'What is the age of # ?'),
    ('per:alternate_names', 'What is the alternate name of # ?'),
    ('per:alumni', 'Who is the alumni of # ?'),
    ('per:boss', 'Who is the employees of # ?'),
    ('per:children', 'Who is parent of # ?'),
    ('per:client', 'Who serves for # ?'),
    ('per:date_of_birth', 'What is the birthday of # ?'),
    ('per:dates', 'Who dates with # ?'),
    ('per:employee_or_member_of', 'What company does # work at ?'),
    ('per:friends', 'Who is the friend of # ?'),
    ('per:girl/boyfriend', 'Who is the girlfriend or boyfriend of # ?'),
    ('per:major', 'What is the major of # ?'),
    ('per:negative_impression', 'Who # has negative impression on ?'),
    ('per:neighbor', 'Who is the neighbor of # ?'),
    ('per:origin', 'Who is the origin of # ?'),
    ('per:other_family', 'Who is the family member of # ?'),
    ('per:parents', 'Who is the child of # ?'),
    ('per:pet', 'What is the pet of # ?'),
    ('per:place_of_residence', 'Where does # live ?'),
    ('per:place_of_work', 'Where does # work ?'),
    ('per:positive_impression', 'Who # has negative impression on ?'),
    ('per:roommate', 'Who is the roommate of # ?'),
    ('per:schools_attended', 'which school does # go to ?'),
    ('per:siblings', 'Who is the sibling of # ?'),
    ('per:spouse', 'Who is the spourse of # ?'),
    ('per:subordinate', 'Who is the subordinate of # ?'),
    ('per:title', 'What is the title of # ?'),
    ('per:visited_place', 'Where does # visit ?'),
    ('per:works', 'Where does # work at ?')
]

import json
def build_qa_data(filename):
    with open(filename) as filein:
        data = json.loads(filein.read())
        for elem in data:
            speakers = []
            for e, t in elem[1]:
                if e.startswith('Speaker'):
                    speakers.append(e)
            # print(speakers)            
            # print(elem[2])

            r_set = {}
            for item in elem[2]:
                if (item[0], item[3]) in r_set:
                    print('MULTIPLE', item[0], item[3])
                r_set[(item[0], item[3])] = item[1]

            question_list = []
            for s in speakers:
                for q in relation_questions:
                    q_r = q[0]
                    question = q[1].replace('#', s)
                    answer = r_set.get((s, q_r), '')
                    question_list.append((question, answer))
            elem.append(question_list)
        return data


def replace_with_slot(x):
    for k in entity_transfer_set:
        x = x.replace(k, entity_transfer_set[k])
    return x

def tokenization(tokenizer, x):
    tokens = []
    for word in x.split():
        if word[0] == '[' and word[-1] == ']':
            tokens.extend([word])
        else:
            tokens.extend(tokenizer.tokenize(word))
    return tokens


def to_bert_example(d, max_seq_length=512):
    dialog = ' '.join(d[0])
    for k in entity_transfer_set:
        dialog = dialog.replace(k+':', entity_transfer_set[k]+' :')

    sub_tokens = tokenization(tokenizer, dialog)
    d_sub_tokens = tokenizer.convert_tokens_to_ids(sub_tokens)

    entity_pos = [] ## each entity position 
    for ent, ent_type in d[1]:
        ent = replace_with_slot(ent)
        sub_tokens = tokenization(tokenizer, ent)
        ent_sub_tokens = tokenizer.convert_tokens_to_ids(sub_tokens)
        res = _find_sub_list_all(ent_sub_tokens, d_sub_tokens)
        entity_pos.append(res)

    all_examples = []

    for q, a in d[-1]:
        q = replace_with_slot(q)
        a = replace_with_slot(a)

        # answer sub tokens
        sub_tokens = tokenization(tokenizer, a)
        a_sub_tokens = tokenizer.convert_tokens_to_ids(sub_tokens)

        res = _find_sub_list_all(a_sub_tokens, d_sub_tokens)

        # question sub tokens
        q = '[CLS] ' + q + ' [SEP]'
        sub_tokens = tokenization(tokenizer, q)
        q_sub_tokens = tokenizer.convert_tokens_to_ids(sub_tokens)
        len_q_sub_tokens = len(q_sub_tokens)

        # print(q_sub_tokens)
        # print(d_sub_tokens)

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

        s, e = res[0]
        if s == -1:
            s = 0
            e = 0
        else:
            s = s + len_q_sub_tokens
            e = s + len_q_sub_tokens
        
        all_examples.append(
            [input_ids, input_mask, segment_ids, s, e]
        )
    return all_examples, entity_pos
    

def to_bert_example_corpus(data, max_seq_length=512):
    all_examples = []
    for d in data:
        t, entity_pos = to_bert_example(d, max_seq_length)
        d.append(entity_pos)
        all_examples.append([d, t])
    return all_examples


train_data = build_qa_data('data/train_mc.json')
train_examples = to_bert_example_corpus(train_data)

dev_data = build_qa_data('data/dev_mc.json')
dev_examples = to_bert_example_corpus(dev_data)

test_data = build_qa_data('data/test_mc.json')
test_examples = to_bert_example_corpus(test_data)

all_examples = [train_examples, dev_examples, test_examples]
with open('data/data.pickle', 'wb') as f:
    pickle.dump(all_examples, f, pickle.HIGHEST_PROTOCOL)

