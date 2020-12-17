import torch

def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(model, path):
    model.load_state_dict(torch.load(path))

def _find_sub_list(sl, l):
    if len(sl) == 0:
        return -1, -1
    results = []
    sll = len(sl)
    for ind in (i for i,e in enumerate(l) if e==sl[0]):
        if l[ind:ind+sll]==sl:
            return ind, ind+sll-1
    return -1, -1


def _find_sub_list_all(sl, l):
    if len(sl) == 0:
        return [(-1, -1)]
    results = []
    sll = len(sl)
    for ind in (i for i,e in enumerate(l) if e==sl[0]):
        if l[ind:ind+sll]==sl:
            results.append((ind, ind+sll-1))
    if len(results) > 0:
        return results
    return [(-1, -1)]


# from transformers import AutoTokenizer, AutoModelForQuestionAnswering

# tokenizer = AutoTokenizer.from_pretrained("deepset/bert-base-cased-squad2")
# model = AutoModelForQuestionAnswering.from_pretrained("deepset/bert-base-cased-squad2")


# tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")
# model = AutoModelForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2")