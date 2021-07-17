import json
import sys
import os
sys.path.append('/home/liuxiaofeng/code/bluebert-master/mt-bert/mt_bert/')
from sklearn.metrics import confusion_matrix
from MicroCalculate import calculateMicroValue
import numpy as np


def writer(write_result, p_dict, r_dict, f1_dict, number_dict):
    p_str = json.dumps(p_dict)
    r_str = json.dumps(r_dict)
    f1_str = json.dumps(f1_dict)
    number_str = json.dumps(number_dict)

    write_result.write(p_str)
    write_result.write('\n')
    write_result.write(r_str)
    write_result.write('\n')
    write_result.write(f1_str)
    write_result.write('\n')
    write_result.write(number_str)
    write_result.write('\n')


def main(data_path, result_path, task_type, write_path, model_type):
    uid = []
    instance_len = []
    y_true = []
    y_pred = []

    y_pred_100 = []
    y_true_100 = []

    y_pred_200 = []
    y_true_200 = []

    y_pred_300 = []
    y_true_300 = []

    y_pred_400 = []
    y_true_400 = []

    y_pred_512 = []
    y_true_512 = []

    with open(result_path, 'r', encoding='utf-8') as result_files:
        for file in result_files.readlines():
            result = json.loads(file)
            y_pred = result['predictions']

    with open(data_path, 'r', encoding='utf-8') as data_files:
        i = 0
        for file in data_files.readlines():
            data = json.loads(file)
            uid.append((data['uid']))
            y_true.append(data["label"])
            length = len(data['token_id'])
            instance_len.append(length)
            if task_type == 'ppim' or task_type == 'cdr':
                if length < 100:
                    y_true_100.append(data['label'])
                    y_pred_100.append(y_pred[i])
                elif length < 200:
                    y_true_200.append(data['label'])
                    y_pred_200.append(y_pred[i])
                elif length < 300:
                    y_true_300.append(data['label'])
                    y_pred_300.append(y_pred[i])
                elif length < 400:
                    y_true_400.append(data['label'])
                    y_pred_400.append(y_pred[i])
                else:
                    y_true_512.append(data['label'])
                    y_pred_512.append(y_pred[i])
            else:
                if length < 100:
                    y_true_100.append(data['label'])
                    y_pred_100.append(y_pred[i])
                else:
                    y_true_200.append(data['label'])
                    y_pred_200.append(y_pred[i])
            i = i + 1

    print(np.mean(instance_len))

    if task_type == 'ddi':
        print('%s task:' % task_type)
        p_dict, r_dict, f1_dict, number_dict = calculateMicroValue(y_true, y_pred, 4)

        matrix = confusion_matrix(y_true, y_pred)
        print(matrix)

        path = os.path.join(write_path, model_type + '_ddi.json')

        write_result = open(path, 'w+', encoding='utf-8')
        write_result.write('The analysis of all result:\n')
        writer(write_result, p_dict, r_dict, f1_dict, number_dict)

        write_result.write(str(matrix))

        if y_pred_100:
            print('\nthe length is 100: \n')
            print(len(y_pred_100))
            p_dict, r_dict, f1_dict, number_dict = calculateMicroValue(y_true_100, y_pred_100, 4)

            write_result.write('\nthe length is 100: \n')
            writer(write_result, p_dict, r_dict, f1_dict, number_dict)

        if y_pred_200:
            print('\nthe length is 200: \n')
            print(len(y_pred_200))
            p_dict, r_dict, f1_dict, number_dict = calculateMicroValue(y_true_200, y_pred_200, 4)

            write_result.write('\nthe length is 200: \n')
            writer(write_result, p_dict, r_dict, f1_dict, number_dict)

        write_result.close()

    elif task_type == 'chemprot':
        print('%s task:' % task_type)
        p_dict, r_dict, f1_dict, number_dict = calculateMicroValue(y_true, y_pred, 5)

        matrix = confusion_matrix(y_true, y_pred)
        print(matrix)

        path = os.path.join(write_path, model_type + '_cpr.json')
        write_result = open(path, 'w+', encoding='utf-8')

        write_result.write('The analysis of all result:\n')
        writer(write_result, p_dict, r_dict, f1_dict, number_dict)
        write_result.write(str(matrix))

        if y_pred_100:
            print('\nthe length is 100: \n')
            print(len(y_pred_100))
            p_dict, r_dict, f1_dict, number_dict = calculateMicroValue(y_true_100, y_pred_100, 5)

            write_result.write('\nthe length is 100: \n')
            writer(write_result, p_dict, r_dict, f1_dict, number_dict)

        if y_pred_200:
            print('\nthe length is 200: \n')
            print(len(y_pred_200))
            p_dict, r_dict, f1_dict, number_dict = calculateMicroValue(y_true_200, y_pred_200, 5)

            write_result.write('\nthe length is 200: \n')
            writer(write_result, p_dict, r_dict, f1_dict, number_dict)

        write_result.close()
    else:
        print('%s task:' % task_type)
        p_dict, r_dict, f1_dict, number_dict = calculateMicroValue(y_true, y_pred, 2)
        matrix = confusion_matrix(y_true, y_pred)
        print(matrix)
        if task_type == 'ppim':
            path = os.path.join(write_path, model_type + '_ppim.json')
        else:
            path = os.path.join(write_path, model_type + '_cdr.json')
        write_result = open(path, 'w+', encoding='utf-8')

        write_result.write('The analysis of all result:\n')
        writer(write_result, p_dict, r_dict, f1_dict, number_dict)
        write_result.write(str(matrix))

        if y_pred_100:
            print('\nthe length is 100: \n')
            p_dict, r_dict, f1_dict, number_dict = calculateMicroValue(y_true_100, y_pred_100, 2)
            write_result.write('\nthe length is 100: \n')
            writer(write_result, p_dict, r_dict, f1_dict, number_dict)

        if y_pred_200:
            print('\nthe length is 200: \n')
            p_dict, r_dict, f1_dict, number_dict = calculateMicroValue(y_true_200, y_pred_200, 2)
            write_result.write('\nthe length is 200: \n')
            writer(write_result, p_dict, r_dict, f1_dict, number_dict)

        if y_pred_300:
            print('\nthe length is 300: \n')
            p_dict, r_dict, f1_dict, number_dict = calculateMicroValue(y_true_300, y_pred_300, 2)
            write_result.write('\nthe length is 300: \n')
            writer(write_result, p_dict, r_dict, f1_dict, number_dict)

        if y_pred_400:
            print('\nthe length is 400: \n')
            p_dict, r_dict, f1_dict, number_dict = calculateMicroValue(y_true_400, y_pred_400, 2)
            write_result.write('\nthe length is 400: \n')
            writer(write_result, p_dict, r_dict, f1_dict, number_dict)

        if y_pred_512:
            print('\nthe length is 512: \n')
            p_dict, r_dict, f1_dict, number_dict = calculateMicroValue(y_true_512, y_pred_512, 2)
            write_result.write('\nthe length is 512: \n')
            writer(write_result, p_dict, r_dict, f1_dict, number_dict)

        write_result.close()


if __name__ == '__main__':
    main('/home/liuxiaofeng/code/bluebert-master/bio_data/canonical_data/bert_uncased_512_lower/chemprot_test.json',
         '/home/liuxiaofeng/code/bluebert-master/checkpoints_sci/sci-mt-dnn_chemprot_adam_5_2e-5_2020-07-27T2027/chemprot_test_scores_4.json',
         'chemprot',
         '/home/liuxiaofeng/code/bluebert-master/checkpoints_sci/analysis/linear_classifier',
         'singletask')
