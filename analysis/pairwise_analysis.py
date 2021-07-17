import json
import sys
from MicroCalculate import calculateMicroValue
from pytorch_pretrained_bert.tokenization import BertTokenizer
import numpy as np


def analysis_pairwise_num(result_path, data_path):
    pair_count = {}
    pair_list = []
    pair_count_id = {}

    pred_pair_5 = []
    true_pair_5 = []

    pred_pair_10 = []
    true_pair_10 = []

    pred_pair_11 = []
    true_pair_11 = []

    with open(result_path, 'r', encoding='utf-8') as result_files:
        for file in result_files.readlines():
            result = json.loads(file)
            y_pred = result['predictions']

    with open(data_path, 'r', encoding='utf-8') as data_files:
        for i, data in enumerate(data_files.readlines()):
            document_id, label_text, text = data.split('\t')
            if label_text == 'positive':
                label = 0
            else:
                label = 1

            token_fn = BertTokenizer.from_pretrained('/data1/home/liuxiaofeng/code/bluebert-master/model/scibert_uncased/vocab.txt', do_lower_case=True)

            text_token = token_fn.tokenize(text)[:512]

            e1_idx = []
            e2_idx = []

            for j, token in enumerate(text_token):
                if token == 'chemical':
                    e1_idx.append(j)
                elif token == 'disease':
                    e2_idx.append(j)

            pair_num = len(e1_idx) * len(e2_idx)
            pair_list.append(pair_num)
            if pair_num not in pair_count:
                pair_count[pair_num] = 1
                pair_count_id[pair_num] = [document_id]
            else:
                pair_count[pair_num] += 1
                pair_count_id[pair_num].append(document_id)

            if pair_num <= 5:
                pred_pair_5.append(y_pred[i])
                true_pair_5.append(label)
            elif pair_num <= 10:
                pred_pair_10.append(y_pred[i])
                true_pair_10.append(label)
            else:
                pred_pair_11.append(y_pred[i])
                true_pair_11.append(label)

        pair_count = sorted(pair_count.items(), key=lambda x: x[0])
        pair_count_id = sorted(pair_count_id.items(), key=lambda x: x[0])
        pair_list = sorted(pair_list)

        print(pair_count)
        # print(dict(pair_count_id)[6])
        # print(pair_list[int(i / 3)])

        count = 0
        for label in true_pair_5:
            if label == 0:
                count += 1
        print(count)

        p_dict, r_dict, f1_dict, number_dict = calculateMicroValue(true_pair_5, pred_pair_5, 2)
        print('\n')
        p_dict, r_dict, f1_dict, number_dict = calculateMicroValue(true_pair_10, pred_pair_10, 2)
        print('\n')
        p_dict, r_dict, f1_dict, number_dict = calculateMicroValue(true_pair_11, pred_pair_11, 2)
        print('\n')
        p_dict, r_dict, f1_dict, number_dict = calculateMicroValue(true_pair_5+true_pair_10+true_pair_11,
                                                                   pred_pair_5+pred_pair_10+pred_pair_11, 2)


def analysis_precision(result1_path, result2_path, data_path):
    pair_count = {}
    pair_list = []
    pair1_pred_id = {}
    pair2_pred_id = {}
    pair1_wrong_id = {}
    pair2_wrong_id = {}

    with open(result1_path, 'r', encoding='utf-8') as result1_files:
        for file in result1_files.readlines():
            result1 = json.loads(file)
            y_pred_1 = result1['predictions']

    with open(result2_path, 'r', encoding='utf-8') as result2_files:
        for file in result2_files.readlines():
            result2 = json.loads(file)
            y_pred_2 = result2['predictions']

    with open(data_path, 'r', encoding='utf-8') as data_files:
        for i, data in enumerate(data_files.readlines()):
            document_id, label_text, text = data.split('\t')
            if label_text == 'positive':
                label = 0
            else:
                label = 1

            token_fn = BertTokenizer.from_pretrained('/data1/home/liuxiaofeng/code/bluebert-master/model/scibert_uncased/vocab.txt', do_lower_case=True)

            text_token = token_fn.tokenize(text)[:512]

            e1_idx = []
            e2_idx = []

            for j, token in enumerate(text_token):
                if token == 'chemical':
                    e1_idx.append(j)
                elif token == 'disease':
                    e2_idx.append(j)

            pair_num = len(e1_idx) * len(e2_idx)
            pair_list.append(pair_num)
            if pair_num not in pair_count:
                pair_count[pair_num] = 1

            else:
                pair_count[pair_num] += 1

            if pair_num not in pair1_pred_id:
                if label == 0 and y_pred_1[i] == 0:
                    pair1_pred_id[pair_num] = [document_id]
            else:
                if label == 0 and y_pred_1[i] == 0:
                    pair1_pred_id[pair_num].append(document_id)

            if pair_num not in pair2_pred_id:
                if label == 0 and y_pred_2[i] == 0:
                    pair2_pred_id[pair_num] = [document_id]
            else:
                if label == 0 and y_pred_2[i] == 0:
                    pair2_pred_id[pair_num].append(document_id)

            # wrong cases
            if pair_num not in pair1_wrong_id:
                if label == 0 and y_pred_1[i] == 1:
                    pair1_wrong_id[pair_num] = [document_id]
            else:
                if label == 0 and y_pred_1[i] == 1:
                    pair1_wrong_id[pair_num].append(document_id)

            if pair_num not in pair2_wrong_id:
                if label == 0 and y_pred_2[i] == 1:
                    pair2_wrong_id[pair_num] = [document_id]
            else:
                if label == 0 and y_pred_2[i] == 1:
                    pair2_wrong_id[pair_num].append(document_id)

        pair_count = sorted(pair_count.items(), key=lambda x: x[0])
        print(pair_count)
        pair1_pred6_id = set(pair1_pred_id[2])
        pair2_pred6_id = set(pair2_pred_id[2])
        # print(pair1_pred6_id.intersection(pair2_pred6_id), '\n')
        #
        # print(pair1_pred6_id.difference(pair2_pred6_id), '\n')
        # print(pair2_pred6_id.difference(pair1_pred6_id), '\n')

        pair1_wrong_case = set(pair1_wrong_id[2])
        pair2_wrong_case = set(pair2_wrong_id[2])

        print('pair wise 1 wrong', len(pair1_wrong_id[1]), pair1_wrong_id[1])
        print('pair wise 2 wrong', len(pair1_wrong_id[2]), pair1_wrong_id[2])
        print('pair wise 4 wrong', len(pair1_wrong_id[4]), pair1_wrong_id[4])
        print('pair wise 4 wrong', len(pair1_wrong_id[6]), pair1_wrong_id[6])

        # print('pair wise 2 pair1 right pair2 wrong', set(pair1_pred_id[2]).intersection(set(pair2_wrong_id[2])))
        # print('pair wise 4 pair1 right pair2 wrong', set(pair1_pred_id[4]).intersection(set(pair2_wrong_id[4])))

        # print('pair wise 1 pair1 wrong pair2 wrong', set(pair1_wrong_id[1]).intersection(set(pair2_wrong_id[1])))
        # print('pair wise 2 pair1 wrong pair2 wrong', set(pair1_wrong_id[2]).intersection(set(pair2_wrong_id[2])))
        # print('pair wise 4 pair1 wrong pair2 wrong', set(pair1_wrong_id[4]).intersection(set(pair2_wrong_id[4])))
        # print(pair1_wrong_case.intersection(pair2_wrong_case), '\n')


if __name__ == '__main__':
    # analysis_pairwise_num(
    # '/data1/home/liuxiaofeng/code/bluebert-master/checkpoints_engineering/sci-bilstm+biaffine+cnn_cdr-document_adam_5_2e-5_2020-11-28T1010/cdr-document_test_scores_4.json',
    #          '/data1/home/liuxiaofeng/code/bluebert-master/bio_data/canonical_data/cdr-document_test.tsv')

    # analysis_pairwise_num(
    #     '/data1/home/liuxiaofeng/code/bluebert-master/checkpoints/sci-cgi+bilstm_cdr-document_adam_5_2e-5_2020-12-12T1724/cdr-document_test_scores_3.json',
    #     '/data1/home/liuxiaofeng/code/bluebert-master/bio_data/canonical_data/cdr-document_test.tsv')

    # analysis_pairwise_num(
    #     '/data1/home/liuxiaofeng/code/bluebert-master/checkpoints/sci-egi_cdr-document_adam_5_2e-5_2020-11-14T1429/cdr-document_test_scores_2.json',
    #     '/data1/home/liuxiaofeng/code/bluebert-master/bio_data/canonical_data/cdr-document_test.tsv')

    analysis_precision('/data1/home/liuxiaofeng/code/bluebert-master/checkpoints_engineering/sci-bilstm+biaffine+cnn_cdr-document_adam_5_2e-5_2020-11-28T1010/cdr-document_test_scores_4.json',
                       '/data1/home/liuxiaofeng/code/bluebert-master/checkpoints/sci-egi_cdr-document_adam_5_2e-5_2020-11-14T1429/cdr-document_test_scores_2.json',
                       '/data1/home/liuxiaofeng/code/bluebert-master/bio_data/canonical_data/cdr-document_test.tsv')