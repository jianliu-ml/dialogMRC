import os
import pickle
import torch
import random
import config
import pickle

from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

from graph_dataset import Dataset
from util import save_model, load_model

from graph_model import BertForQuestionAnswering as QAModel
# from graph_model import RobertaForQuestionAnswering as QAModel


cuda_list = '2,3'
os.environ['CUDA_VISIBLE_DEVICES'] = cuda_list

## redefin_load_model
def load_model_from_dir(model_path, device):
    model = QAModel.from_pretrained(model_path).to(device)
    model.rgcn.to(device)
    return model


def filter_neg(train_examples):
    train_all = []
    for elem in train_examples:
        all_text, bert_exps = elem
        for e in bert_exps:
            question, qa_id, ans, bert_feature, map_to_origin, len_q_sub_tokens, res = e
            train_all.append(bert_feature + [res])
    return train_all


if __name__ == '__main__':
    # parameters
    n_epoch = 50
    batch_size = 4 * (len(cuda_list.split(',')) + 1)

    learning_rate = 5e-5
    adam_epsilon = 1e-8
    warmup_steps = 0
    max_grad_norm = 1.0

    # load model
    device = 'cuda'
    # model = load_model_from_dir(config.bert_dir, device)
    model = load_model_from_dir('/data/jliu/research/DialogMC/squad_base_cased_finetuned', device)
    # model = load_model_from_dir('deepset/bert-base-cased-squad2', device)
    # model = load_model_from_dir("deepset/roberta-base-squad2", device)
    model = torch.nn.DataParallel(model)

    # load_model(model, 'model/1.pt')

    with open('data/data.pickle', 'rb') as f:
        [train_examples, dev_examples, test_examples] = pickle.load(f)

    training_set = filter_neg(train_examples)
    print(len(training_set))
    train_dataset = Dataset(batch_size, training_set)

    t_total = int(n_epoch * len(training_set) / batch_size)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, t_total)
    
    global_step = 0
    evaluate_every = 200
    for idx in range(n_epoch):
        for batch in train_dataset.get_tqdm(device, shuffle=True):
            global_step += 1
            model.train()
            input_ids, input_mask, segment_ids, start_positions, end_positions, graphs, edge_types = batch

            inputs = {'input_ids':      input_ids,
                    'attention_mask':   input_mask,
                    'token_type_ids':   segment_ids,
                    'start_positions':  start_positions,
                    'end_positions':    end_positions,
                    'graphs':           graphs,
                    'edge_types':       edge_types}

            outputs = model(**inputs)
            loss = outputs[0]

            #
            loss = loss.mean()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()
            scheduler.step() 
            model.zero_grad()
        
            # if global_step % evaluate_every == 0:
            #     model.eval()
            #     with torch.no_grad():
            #         model_evaluation(model, dev_dataset, device)
            #         # model_evaluation(model, framenet_test_dataset, device)
        
        save_model(model, 'model/%d.pt' % idx)

        # model.eval()
        # with torch.no_grad():
        #     model_evaluation(model, dev_dataset, device)t5g6