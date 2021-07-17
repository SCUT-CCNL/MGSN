#_*_coding:utf-8_*_


def calculateMicroValue(y_true, y_pred, label_num):
    all_label_result = dict()

    for label in range(label_num):
        all_label_result[label] = calculateF(y_true, y_pred, label)
    tp_all = fp_all = fn_all = 0.0
    p_dict = dict()
    r_dict = dict()
    f1_dict = dict()
    number_dict = dict()
    for label in all_label_result:
        tp = all_label_result[label][0]
        fp = all_label_result[label][1]
        fn = all_label_result[label][2]
        tp_all += tp
        fp_all += fp
        fn_all += fn
        p = tp / (tp + fp + 0.00001)
        r = tp / (tp + fn + 0.00001)
        f1 = 2 * p * r / (p + r + 0.00001)
        p_dict[label] = p
        r_dict[label] = r
        f1_dict[label] = f1
        number_dict[label] = tp + fn
        print('label:', label, ' precision: ', p, ' recall: ', r, ' f1_score: ', f1, ' numbers: ', tp + fn)
    p_all = tp_all / (tp_all + fp_all + 0.00001)
    r_all = tp_all / (tp_all + fn_all + 0.00001)
    f1_all = 2 * p_all * r_all / (p_all + r_all + 0.00001)
    p_dict['all_labels'] = p_all
    r_dict['all_labels'] = r_all
    f1_dict['all_labels'] = f1_all
    number_dict['all_labels'] = tp_all + fn_all
    print('all_label', ' precision: ', p_all, ' recall: ', r_all, ' f1_score: ', f1_all, ' numbers: ', tp_all + fn_all)
    return p_dict, r_dict, f1_dict, number_dict


def calculateF(y_true, y_pred, label):
    tp = fp = fn = 0.0
    for left, right in zip(y_pred, y_true):
        if left == label and right == label:
            tp += 1
        if left == label and right != label:
            fp += 1
        if left != label and right == label:
            fn += 1
    return tp, fp, fn



# import numpy
# y_pred = numpy.random.randint(low=0, high=3, size=(5, ))
# print(y_pred)
# y_true = numpy.random.randint(low=0, high=3, size=(5, ))
# print(y_true)
# p_dict, r_dict, f1_dict, number = calculateMicroValue(y_pred, y_true, [0, 1])

