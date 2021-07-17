import numpy as np
import re
from spacy.lang.en import English
import spacy
from tqdm import tqdm
import json


def data_preprocessing(raw_text):
    text = re.sub(r'\(.{1,20}\)', '', raw_text)
    # print(text)
    nlp = spacy.load('en')
    doc = nlp(text)
    filter_words = set()
    all_words = list()
    # pos = ['ADJ', 'ADV', 'AUX', 'CONJ', 'NOUN', 'PROPN', 'CCONJ', 'SCONJ', 'VERB']
    pos = ['VERB', 'NOUN']
    for token in doc:
        all_words.append(token.lemma_)
        if token.text.startswith('@'):
            continue
        if token.text == 'DRUG1' or token.text == 'DRUG2' or token.text == 'DRUG0':
            continue
        if token.text == 'chemical' or token.text == 'disease':
            continue
        if token.text == 'CHEMICAL' or token.text == 'DISEASE':
            continue
        if token.text == 'substance' or token.text == 'mention':
            continue
        if token.text == 'GENE1' or token.text == 'GENE2':
            continue
        if re.match(r'D\d{5}', token.text):
            continue

        if token.pos_ in pos:
            # print(token.text, token.lemma_, token.pos_, token.tag_, token.is_stop)
            filter_words.add(token.lemma_)
            # filter_words.add(token.text)
        # print(token.text, token.lemma_, token.pos_, token.tag_, token.is_stop)
    # with open('/home/liuxiaofeng/code/bluebert-master/bio_data/ppim-2_corpus.txt', 'a+', encoding='utf-8') as f:
    #     sentence = ' '.join(all_words)
    #     f.write(sentence + '\n')
    return list(filter_words)


def get_data(file_path, label_dict):
    terms = {}
    n = len(label_dict)
    label_num = [0] * n

    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(tqdm((f.readlines()))):
            words = []
            if i == 0:
                continue
            index, sentence, label = line.strip('\n').split('\t')

            label_id = label_dict[label]
            if file_path.find('chemprot') != -1 or file_path.find('ddi') != -1:
                sentence = regularization(sentence)
                words = data_preprocessing(sentence)
            elif file_path.find('sentences') != -1:
                words = data_preprocessing(sentence)
            elif file_path.find('ppim') != -1:
                s_list = sentence.split('. ')
                for s in s_list:
                    s = regularization(s)
                    words.extend(data_preprocessing(s))
            else:
                s_list = sentence.split(' . ')
                for s in s_list:
                    words.extend(data_preprocessing(s))

            words = list(set(words))

            for w in words:
                if w == '' or w == ' ':
                    continue
                if w not in terms:
                    terms[w] = [0] * n
                terms[w][label_id] += 1
            label_num[label_id] += 1
    return terms, label_num


def regularization(sentence):
    sentence = sentence.replace('(', ' ( ')
    sentence = sentence.replace(')', ' ) ')
    sentence = sentence.replace('-', ' - ')
    sentence = sentence.replace('/', ' / ')
    # sentence = sentence.replace('.', ' . ')
    sentence = sentence.replace(',', ' , ')
    sentence = sentence.replace('\"', ' " ')
    sentence = sentence.replace(':', ' : ')
    sentence = sentence.replace('\'', ' \' ')
    sentence = sentence.replace(';', ' ; ')
    sentence = sentence.replace('[', ' [ ')
    sentence = sentence.replace(']', ' ] ')
    sentence = sentence.replace('\\', ' \\ ')
    sentence = sentence.replace('=', ' = ')
    sentence = sentence.replace('<', ' < ')
    sentence = sentence.replace('>', ' > ')
    sentence = sentence.replace('   ', ' ')
    sentence = sentence.replace('  ', ' ')
    sentence = ' '.join([w for w in sentence.split(' ') if w != '' or w != ' '])
    return sentence


def chi_score(N11, N10, N01, N00):
    s1 = (N11 + N10 + N01 + N00) * (N11 * N00 - N10 * N01) * (N11 * N00 - N10 * N01)
    s2 = (N11 + N10) * (N11 + N01) * (N10 + N00) * (N01 + N00)
    return s1 / (s2 + 0.00001)


def get_label_chi(label_id, terms, label_num, output):
    chi = {}
    for k in terms:
        N11 = terms[k][label_id]
        N10 = sum(terms[k]) - N11
        N01 = label_num[label_id] - N11
        N00 = sum(label_num) - N01 - N10 - N11
        chi[k] = chi_score(N11, N10, N01, N00)
    chi = dict(sorted(chi.items(), key=lambda k: k[1], reverse=True))

    for k in chi:
        df = terms[k][label_id] / (label_num[label_id] + 0.00001)
        line = k + '\t' + str(chi[k]) + '\t' + str(df) + '\n'
        output.write(line)


def selected_words(file1, file2, label_name, selected_num, threshold):
    word_list = []
    with open(file1, 'r', encoding='utf-8') as r:
        i = 0
        for line in r.readlines():
            word, chi2, df = line.split('\t')
            if float(df) > threshold:
                word_list.append(word)
                i += 1
            if i >= selected_num:
                break
    with open(file2, 'a+', encoding='utf-8') as w:
        w.write(label_name + "\t" + ' '.join(word_list) + '\n')


def get_ddi_chi(ddi_file):
    label_dict = {'int': 0, 'advise': 1, 'effect': 2, 'mechanism': 3, 'other': 4}
    terms, label_num = get_data(ddi_file, label_dict)
    with open('/data1/home/liuxiaofeng/code/bluebert-master/bio_data/ddi/keywords/words_chi_verb.json', 'w', encoding='utf-8') as f:
        json.dump(terms, f)
        f.write('\n')
        json.dump(label_num, f)

    for label_name, label_id in label_dict.items():
        print(label_name)
        output_file = '/data1/home/liuxiaofeng/code/bluebert-master/bio_data/ddi/keywords/' + label_name + '_keyword_verb.txt'
        with open(output_file, 'w', encoding='utf-8') as output:
            get_label_chi(label_id, terms, label_num, output)
        writen_file = '/data1/home/liuxiaofeng/code/bluebert-master/bio_data/ddi/keywords/selected_keywords_verb.txt'
        selected_words(output_file, writen_file, label_name, 15, 0.05)


def get_cpr_chi(cpr_file):
    label_dict = {'CPR:3': 0, 'CPR:4': 1, 'CPR:5': 2, 'CPR:6': 3, 'CPR:9': 4, 'false': 5}
    terms, label_num = get_data(cpr_file, label_dict)

    with open('/data1/home/liuxiaofeng/code/bluebert-master/bio_data/chemprot/keywords/words_chi_verb.json', 'w', encoding='utf-8') as f:
        json.dump(terms, f)
        f.write('\n')
        json.dump(label_num, f)

    for label_name, label_id in label_dict.items():
        print(label_name)
        output_file = '/data1/home/liuxiaofeng/code/bluebert-master/bio_data/chemprot/keywords/' + label_name + '_keyword_verb.txt'
        with open(output_file, 'w', encoding='utf-8') as output:
            get_label_chi(label_id, terms, label_num, output)
        writen_file = '/data1/home/liuxiaofeng/code/bluebert-master/bio_data/chemprot/keywords/selected_keywords_verb.txt'
        selected_words(output_file, writen_file, label_name, 15, 0.05)


def get_cdr_chi(cpr_file):
    label_dict = {'positive': 0, 'negative': 1}
    terms, label_num = get_data(cpr_file, label_dict)

    # with open('/data1/home/liuxiaofeng/code/bluebert-master/bio_data/cdr/keywords/words_chi.json', 'w', encoding='utf-8') as f:
    #     json.dump(terms, f)
    #     f.write('\n')
    #     json.dump(label_num, f)

    for label_name, label_id in label_dict.items():
        print(label_name)
        output_file = '/data1/home/liuxiaofeng/code/bluebert-master/bio_data/cdr/keywords/' + label_name + '_keyword.txt'
        with open(output_file, 'w', encoding='utf-8') as output:
            get_label_chi(label_id, terms, label_num, output)
        writen_file = '/data1/home/liuxiaofeng/code/bluebert-master/bio_data/cdr/keywords/selected_keywords.txt'
        selected_words(output_file, writen_file, label_name, 15, 0.05)


def get_ppim_chi(cpr_file):
    print('ppim')
    label_dict = {'positive': 0, 'negative': 1}
    terms, label_num = get_data(cpr_file, label_dict)

    with open('/data1/home/liuxiaofeng/code/bluebert-master/bio_data/ppim-2/keywords/words_chi_verb.json', 'w', encoding='utf-8') as f:
        json.dump(terms, f)
        f.write('\n')
        json.dump(label_num, f)

    for label_name, label_id in label_dict.items():
        print(label_name)
        output_file = '/data1/home/liuxiaofeng/code/bluebert-master/bio_data/ppim-2/keywords/' + label_name + \
                      '_keyword_verb.txt'
        with open(output_file, 'w', encoding='utf-8') as output:
            get_label_chi(label_id, terms, label_num, output)
        writen_file = '/data1/home/liuxiaofeng/code/bluebert-master/bio_data/ppim-2/keywords/selected_keywords_verb.txt'
        selected_words(output_file, writen_file, label_name, 15, 0.05)


def get_cdr_document_chi(cdr_file):
    print('cdr')
    label_dict = {'positive': 0, 'negative': 1}
    terms, label_num = get_data(cdr_file, label_dict)

    with open('/data1/home/liuxiaofeng/code/bluebert-master/bio_data/cdr-document/keywords/words_chi_sentence.json', 'w', encoding='utf-8') as f:
        json.dump(terms, f)
        f.write('\n')
        json.dump(label_num, f)

    for label_name, label_id in label_dict.items():
        print(label_name)
        output_file = '/data1/home/liuxiaofeng/code/bluebert-master/bio_data/cdr-document/keywords/' + label_name + \
                      '_keyword_sentences.txt'
        with open(output_file, 'w', encoding='utf-8') as output:
            get_label_chi(label_id, terms, label_num, output)
        writen_file = '/data1/home/liuxiaofeng/code/bluebert-master/bio_data/cdr-document/keywords/selected_keywords_sentences.txt'
        selected_words(output_file, writen_file, label_name, 30, 0.05)

# get_ddi_chi('/data1//home/liuxiaofeng/code/bluebert-master/bio_data/ddi/new.tsv')

# get_cpr_chi('/data1//home/liuxiaofeng/code/bluebert-master/bio_data/chemprot/train.tsv')

# get_cdr_chi('/data1//home/liuxiaofeng/code/bluebert-master/bio_data/cdr/train.tsv')

# get_ppim_chi('/data1/home/liuxiaofeng/code/bluebert-master/bio_data/ppim-2/train.tsv')

get_cdr_document_chi('/data1//home/liuxiaofeng/code/bluebert-master/bio_data/cdr-document/sentences.tsv')


# word_list = []
# file1 = '/data1/home/liuxiaofeng/code/bluebert-master/bio_data/ddi/keywords/other_keyword_verb.txt'
# with open(file1, 'r', encoding='utf-8') as r:
#     i = 0
#     for line in r.readlines():
#         word, chi2, df = line.split('\t')
#         if float(df) > 0.05:
#             print(word)
#             i += 1
#         if i >= 20:
#             break
