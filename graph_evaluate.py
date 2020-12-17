import os
import pickle
import torch
import random
import config
import numpy as np
import re
import string

from transformers import BertTokenizer
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

from graph_model import BertForQuestionAnswering as QAModel
# from graph_model import RobertaForQuestionAnswering as QAModel

from graph_dataset import Dataset
from util import save_model, load_model

cuda_list = '7'
os.environ['CUDA_VISIBLE_DEVICES'] = cuda_list


### taken from friendsQA...
def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude or ch == '_')

    def lower(text):
        return text.lower()

    def remove_underline(text):
        return text.replace('_', ' ')

    return remove_underline(white_space_fix(remove_articles(remove_punc(lower(s)))))



def load_model_from_dir(model_path, device):
    model = QAModel.from_pretrained(model_path).to(device)
    return model

question_length = {'What': [469, 1989], 'Who': [379, 894], 'When': [210, 893], 'Where': [346, 1181], 'Why': [306, 2729], 'How': [298, 1817]}

for elem in question_length:
    print(elem, question_length[elem][1] / question_length[elem][0])


def evaluate_temp(t_examples):
    with open('temp.pickle', 'rb') as f:
        [ground_s, ground_e, predicted_s, predicted_e] = pickle.load(f)

    for q_type in question_length:    
        print(q_type)
        n_gold, n_predict, n_correct = 0, 0, 0
        question_set = {}
        for exp, gs, ge, ps, pe in zip(t_examples, ground_s, ground_e, predicted_s, predicted_e):
            all_text, question, qa_id, ans, map_to_origin, len_q_sub_tokens = exp

            qa_type_from_id = qa_id.split('_')[-1]
            if qa_type_from_id == 'Paraphrased':
                qa_type_from_id = qa_id.split('_')[-2]
            # print(qa_type_from_id)

            if qa_type_from_id != q_type:
                continue

            question_set.setdefault(question, {})
            question_set[question].setdefault('gold', [])
            ans = list(ans)
            ans[2] = normalize_answer(ans[2])
            question_set[question]['gold'].append(ans)

            question_set[question].setdefault('pred', [])

            s_sort = np.argsort(-ps)
            e_sort = np.argsort(-pe)

            q_word = question.split()[0]

            if q_word in question_length:
                max_answer_length = question_length[q_word][1] / question_length[q_word][0] + 3
            else:
                max_answer_length = 8

            res = []
            for s in s_sort[:10]:
                for e in e_sort[:10]:
                    if s <= e and e - s <= max_answer_length:
                        res.append([s, e, ps[s] * pe[e]])
            res = sorted(res, key=lambda x: x[2], reverse=True)
            if len(res) == 0:
                s, e = 0, 0
            else:
                s, e = res[0][0], res[0][1]

            # print(gs, ge, s, e)  # s, e, should substract len_q

            s -= len_q_sub_tokens
            e -= len_q_sub_tokens

            s = map_to_origin[s] if s < len(map_to_origin) else 0
            e = map_to_origin[e] if e < len(map_to_origin) else 0

            # print(ans, s, e)
            predicted_ans = ' '.join(all_text[s:e+1])
            predicted_ans = normalize_answer(predicted_ans)
            question_set[question]['pred'].append(predicted_ans)

        ## normal f1
        n_gold = len(question_set)
        for elem in question_set:
            gold = set([x[2] for x in question_set[elem]['gold']])
            pred = question_set[elem]['pred']
            # print(elem, gold, pred)
            if len(''.join(pred)) > 0:
                n_predict += 1
            for p in pred:
                if p in gold:
                    n_correct += 1
                    break

        p = n_correct / n_predict
        r = n_correct / n_gold
        f1 = 2 * p * r / (p + r)
        print(n_gold, n_predict, n_correct, p, r, f1)

        ## UM
        um = 0
        for elem in question_set:
            temp = []
            for gold in [x[2] for x in question_set[elem]['gold']]:
                for pred in question_set[elem]['pred']:
                    gold_set = set(gold.split())
                    pred_set = set(pred.split())
                    
                    c = gold_set.intersection(pred_set)
                    p = len(c) / (len(pred_set) + 1e-200)
                    r = len(c) / (len(gold_set) + 1e-200)
                    f1 = 2 * p * r / (p + r + 1e-200)
                    temp.append(f1)
            um += max(temp)
        print(um / len(question_set))


if __name__ == '__main__':

    with open('data/data.pickle', 'rb') as f:
        [train_examples, dev_examples, test_examples] = pickle.load(f)
    
    all_qas = []
    test_all = []
    for elem in dev_examples:
        all_text, bert_exps = elem
        for e in bert_exps:
            question, qa_id, ans, bert_feature, map_to_origin, len_q_sub_tokens, res = e
            test_all.append(bert_feature + [res])
            all_qas.append([all_text, question, qa_id, ans, map_to_origin, len_q_sub_tokens])

    evaluate_temp(all_qas)

    print(len(test_all))      

    batch_size = 2 * (len(cuda_list.split(',')) + 1)
    test_set = Dataset(batch_size, test_all)

    # load model
    device = 'cuda'
    model = load_model_from_dir(config.bert_dir, device)
    # model = load_model_from_dir("deepset/roberta-base-squad2", device)
    model = torch.nn.DataParallel(model)

    load_model(model, 'model/best_models/1.pt')
    model.eval()

    ground_s = list()
    ground_e = list()
    predicted_s = list()
    predicted_e = list()

    with torch.no_grad():
        for batch in test_set.get_tqdm(device, shuffle=False):

            input_ids, input_mask, segment_ids, start_positions, end_positions, graphs, edge_types = batch

            inputs = {'input_ids':      input_ids,
                    'attention_mask':   input_mask,
                    'token_type_ids':   segment_ids,
                    'graphs':           graphs,
                    'edge_types':       edge_types}
            outputs = model(**inputs)
            
            predicted_s.extend(torch.softmax(outputs[0], dim=-1).cpu().numpy())
            predicted_e.extend(torch.softmax(outputs[1], dim=-1).cpu().numpy())

            ground_s.extend(start_positions.cpu().numpy())
            ground_e.extend(end_positions.cpu().numpy())


    predict_result = [ground_s, ground_e, predicted_s, predicted_e]
    with open('temp.pickle', 'wb') as f:
        pickle.dump(predict_result, f, pickle.HIGHEST_PROTOCOL)

    evaluate_temp(all_qas)
