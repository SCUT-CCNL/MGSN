import torch
import argparse
import itertools
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
sys.path.append('../mt-bert/mt_bert')
from mt_dnn.model import MTDNNModel
from data_utils.task_def import TaskType, DataFormat
from blue_prepro_std import bert_feature_extractor
from mt_dnn.batcher import BatchGen
from pytorch_pretrained_bert.tokenization import BertTokenizer
from module.biaffine_pairwise import BiaffinePairwiseScore, BiaffinePairwiseScore_3


def arg_config() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='./checkpoints/')
    parser.add_argument('--task', default='cdr-document')
    parser.add_argument('--vocab_file', default='/data1/home/liuxiaofeng/code/bluebert-master/model/scibert_uncased/vocab.txt')
    parser = parser.parse_args()
    return parser


def get_entity_index(premise, task_name, tokenizer, max_seq_len):

    if task_name.find('ddi') != -1:
        e1 = 'DRUG1'
        e2 = 'DRUG2'
    elif task_name.find('cdr-document') != -1:
        e1 = 'CHEMICAL'
        e2 = 'DISEASE'
    elif task_name.find('cdr-test') != -1:
        e1 = 'CHEMICAL'
        e2 = 'DISEASE'
    elif task_name.find('chemprot') != -1:
        e1 = '@CHEMICAL$'
        e2 = '@GENE$'
    elif task_name.find('cdr') != -1:
        e1 = 'chemical'
        e2 = 'disease'
    elif task_name.find('ppim') != -1:
        e1 = 'GENE1'
        e2 = 'GENE2'
    elif task_name.find('chr') != -1:
        e1 = 'CHEMICAL1'
        e2 = 'CHEMICAL2'
    else:
        return None, None

    text = tokenizer.tokenize(premise)
    if len(text) > max_seq_len - 2:
        text = text[:max_seq_len-2]
    text = ['CLS'] + text + ['SEP']
    e1_token = tokenizer.tokenize(e1)
    e2_token = tokenizer.tokenize(e2)

    e1_len = len(e1_token)
    e2_len = len(e2_token)
    text_len = len(text)

    e1_dist = [0] * text_len
    e2_dist = [0] * text_len

    flag = False  # if e2_token not exist
    for i, word in enumerate(text):
        if i + e1_len - 1 <= text_len:
            if ''.join(e1_token) == ''.join(text[i:i + e1_len]):
                e1_dist[i:i + e1_len] = [1] * e1_len
        if i + e2_len - 1 <= text_len:
            if ''.join(e2_token) == ''.join(text[i:i + e2_len]):
                e2_dist[i:i + e2_len] = [1] * e2_len
                flag = True
    if not flag:
        e2_dist = e1_dist

    return e1_dist, e2_dist


def build_data_premise_only(text, task_name, max_seq_len=512, tokenizer=None, task_type=None,):
    """Build data of single sentence tasks
    """

    ids = 0
    premise = text
    label = 0
    input_ids, _, type_ids = bert_feature_extractor(premise, max_seq_length=max_seq_len, tokenize_fn=tokenizer)
    if task_type == TaskType.RelationExtraction:
        e1_dist, e2_dist = get_entity_index(premise, task_name, tokenizer, max_seq_len)
        features = {'uid': ids, 'label': label, 'token_id': input_ids, 'type_id': type_ids, 'e1_dist': e1_dist,
                    'e2_dist': e2_dist}
    else:
        features = {'uid': ids, 'label': label, 'token_id': input_ids, 'type_id': type_ids}

    return [features]


def hook(module, fea_in, fea_out):
    print("hooker working")
    module_name.append(module.__class__)
    features_in_hook.append(fea_in)
    features_out_hook.append(fea_out)
    return None


if __name__ == '__main__':
    args = arg_config()
    state_dict = torch.load(args.model_path)
    config = state_dict['config']
    config['multi_gpu_on'] = False


    x = 'CHEMICAL - associated DISEASE . Can it be treated with bromocriptine ? Six stable psychiatric outpatients with DISEASE and amenorrhea / oligomenorrhea associated with their CHEMICAL were treated with bromocriptine . Daily dosages of 5 - 10 mg corrected the DISEASE and restored menstruation in four of the six patients . One woman , however , developed worsened psychiatric symptoms while taking bromocriptine , and it was discontinued . Thus , three of six patients had their menstrual irregularity successfully corrected with bromocriptine . This suggests that bromocriptine should be further evaluated as potential therapy for CHEMICAL - associated DISEASE and amenorrhea / galactorrhea .'
    token_fn = BertTokenizer.from_pretrained(args.vocab_file, do_lower_case=True)

    feature = build_data_premise_only(x, args.task, tokenizer=token_fn, task_type=TaskType.RelationExtraction)
    visual_data = BatchGen(feature, batch_size=1, dropout_w=config['dropout_w'],  is_train=False, gpu=config['cuda'],
                           task_id=0, maxlen=config['max_seq_len'], data_type=DataFormat.PremiseOnly,
                           task_type=TaskType.RelationExtraction)

    model = MTDNNModel(config, state_dict=state_dict)

    module_name = []
    features_in_hook = []
    features_out_hook = []
    for i, children in enumerate(model.network.modules()):
        if isinstance(children, BiaffinePairwiseScore_3):
            children.register_forward_hook(hook=hook)
        if isinstance(children, BiaffinePairwiseScore):
            children.register_forward_hook(hook=hook)
    for batch_meta, batch_data in visual_data:
        score, pred, gold = model.predict(batch_meta, batch_data)
        # print(score, pred, gold)
    # print(features_in_hook)
    # print(features_out_hook)
    print(token_fn.tokenize(x)[:512])

    x_token = ['CLS'] + token_fn.tokenize(x) + ['SEP']

    e1_idx = []
    e2_idx = []
    for i, word in enumerate(x_token):
        if word == 'chemical':
            e1_idx.append(i)
        elif word == 'disease':
            e2_idx.append(i)
    print(e1_idx, e2_idx)
    pair_weight = torch.FloatTensor(len(e1_idx)*len(e2_idx)).fill_(-1e-8)
    k = 0
    for i in e1_idx:
        for j in e2_idx:
            pair_weight[k] = features_out_hook[0][1][0, i, j, 0]
            k += 1
            print(i, x_token[i:i+2], j, x_token[j:j+2], features_out_hook[0][1][0, i, j])
    print(features_out_hook[0][0])
    print(pair_weight)
    print(torch.nn.functional.softmax(pair_weight, 0))
