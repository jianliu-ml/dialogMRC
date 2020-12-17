import os
import pickle
import torch
import random
import config
import pickle

from transformers import BertTokenizer, BertForQuestionAnswering
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

from dataset import Dataset
from util import save_model, load_model

cuda_list = '7'
os.environ['CUDA_VISIBLE_DEVICES'] = cuda_list


## redefin_load_model
def load_model_from_dir(model_path, device):
    model = BertForQuestionAnswering.from_pretrained(model_path).to(device)
    return model


def filter_neg(examples, filter_ratio=0):
    res = []
    for elem in examples:
        original_emp, examples = elem
        for exp in examples:
            if exp[-1] != 0 or random.random() > filter_ratio:
                res.append(exp)
    return res


if __name__ == '__main__':
    # parameters

    n_epoch = 50
    batch_size = 8 * (len(cuda_list.split(',')) + 1)

    learning_rate = 5e-5
    adam_epsilon = 1e-8
    warmup_steps = 0
    max_grad_norm = 1.0


    # load model
    device = 'cuda'
    model = load_model_from_dir(config.bert_dir, device)
    model = torch.nn.DataParallel(model)

    load_model(model, 'model/old_models/36.pt')

    with open('data/data.pickle', 'rb') as f:
        [train_examples, dev_examples, test_examples] = pickle.load(f)


    training_set = filter_neg(train_examples, 0.5)
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
            input_ids, input_mask, segment_ids, start_positions, end_positions = batch

            inputs = {'input_ids': input_ids,
                    'attention_mask':  input_mask,
                    'token_type_ids':  segment_ids,
                    'start_positions': start_positions,
                    'end_positions':   end_positions}
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
            #         model_evaluation(model, test_dataset, device)
            #         # model_evaluation(model, framenet_test_dataset, device)
        
        save_model(model, 'model/%d.pt' % idx)
        # model.eval()
        # with torch.no_grad():
        #     model_evaluation(model, test_dataset, device)t5g6