import os
import pickle
import torch
import random
import config
import numpy as np

from transformers import BertTokenizer, BertForQuestionAnswering
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

from dataset import Dataset
from util import save_model, load_model

cuda_list = '2,3'
os.environ['CUDA_VISIBLE_DEVICES'] = cuda_list

def load_model_from_dir(model_path, device):
    model = BertForQuestionAnswering.from_pretrained(model_path).to(device)
    return model


def evaluate_temp(test_examples):
    with open('temp.pickle', 'rb') as f:
        [ground_s, ground_e, predicted_s, predicted_e] = pickle.load(f)
    
    n_gold, n_predict, n_correct = 0, 0, 0

    idx_s, idx_e = 0, 0
    for elem in test_examples: # enumerate every example
        idx_e = idx_s + len(elem[1])
        gs, ge, ps, pe = ground_s[idx_s:idx_e], ground_e[idx_s:idx_e], predicted_s[idx_s:idx_e], predicted_e[idx_s:idx_e]
        _, ents, relations, questions, ent_poses = elem[0]
        for qa, bert_example, logit_s, logit_e in zip(questions, elem[1], ps, pe):
            q_len = bert_example[0].index(102)
            q, a = qa 
            a_pred = ''

            zero_prob = logit_s[0] + logit_e[0]
            ent_prob = []
            for ent, ent_pos in zip(ents, ent_poses):
                # print(ent, ent_pos)
                prob = 0
                for ep in ent_pos:
                    ep_s, ep_e = ep[0] + q_len, ep[1] + q_len

                    if ep_e >= 512:
                        continue

                    for i in range(ep_s, ep_e + 1):
                        prob += logit_s[i]
                        prob += logit_e[i]
                ent_prob.append(prob)

            a_pred = ''
            p = ent_prob[np.argmax(ent_prob)]
            if p > 0.1:
                a_pred = ents[np.argmax(ent_prob)][0]


            if a or a_pred:
                print(q, a + ' | ' + a_pred)
            if len(a) > 0:
                n_gold += 1
            if len(a_pred) > 0:
                n_predict += 1
            if len(a) > 0 and a == a_pred:
                n_correct += 1

        idx_s = idx_e
    
    p = n_correct / n_predict
    r = n_correct / n_gold
    f1 = 2 * p * r / (p + r)
    print(n_gold, n_predict, n_correct, p, r, f1)



if __name__ == '__main__':

    with open('data/data.pickle', 'rb') as f:
        [train_examples, dev_examples, test_examples] = pickle.load(f)


    # evaluate_temp(test_examples)
    
    # parameters
    batch_size = 40 * (len(cuda_list.split(',')) + 1)

    # load model
    device = 'cuda'
    model = load_model_from_dir('/data/jliu/research/DialogMC/squad_base_cased_finetuned', device)
    model = torch.nn.DataParallel(model)

    load_model(model, 'model/22.pt')
    model.eval()

    test_set = []
    for elem in test_examples:
        test_set.extend(elem[1])
    
    test_set = Dataset(batch_size, test_set)

    ground_s = list()
    ground_e = list()
    predicted_s = list()
    predicted_e = list()

    with torch.no_grad():
        for batch in test_set.get_tqdm(device, shuffle=False):

            input_ids, input_mask, segment_ids, start_positions, end_positions = batch

            inputs = {'input_ids': input_ids,
                    'attention_mask':  input_mask,
                    'token_type_ids':  segment_ids}
            outputs = model(**inputs)
            
            predicted_s.extend(torch.softmax(outputs[0], dim=-1).cpu().numpy())
            predicted_e.extend(torch.softmax(outputs[1], dim=-1).cpu().numpy())

            ground_s.extend(start_positions.cpu().numpy())
            ground_e.extend(end_positions.cpu().numpy())


    predict_result = [ground_s, ground_e, predicted_s, predicted_e]
    with open('temp.pickle', 'wb') as f:
        pickle.dump(predict_result, f, pickle.HIGHEST_PROTOCOL)

    evaluate_temp(test_examples)