import config
import json
import pickle
import random

from transformers import AutoTokenizer as Tokenizer

from util import _find_sub_list

def adv_friends(filename, filename_adv):
    with open(filename) as filein:
        data = json.loads(filein.read())
        for elem in data['data']:
            utterances = elem['paragraphs']
            utrs = utterances[0]['utterances:']
            qas = utterances[0]['qas']

            idx = random.randint(0, len(utrs) -1 )
            temp = utrs[idx]

            idx = random.randint(0, len(utrs) -1)
            temp_person = utrs[idx]['speakers'][0]

            for qa in qas:
                qa_id = qa['id']
                question = qa['question']
                
                qa_type_from_id = qa_id.split('_')[-1]
                if qa_type_from_id == 'Paraphrased':
                    qa_type_from_id = qa_id.split('_')[-2]
                if qa_type_from_id == 'Who':
                    temp['uid'] = len(utrs)
                    temp['utterance'] = temp_person + ' ' + ' '.join(question.split()[1:-1])
                    elem['paragraphs'][0]['utterances:'].append(temp)
                    break
    

        with open(filename_adv, 'w') as fileout:
            json.dump(data, fileout)


if __name__ == '__main__':
    adv_friends('data/FriendsQA/dat/friendsqa_tst.json', 'data/FriendsQA/dat/friendsqa_tst_adv.json')
    adv_friends('data/FriendsQA/dat/friendsqa_dev.json', 'data/FriendsQA/dat/friendsqa_dev_adv.json')




    