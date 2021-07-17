import torch
import os
import math
import torch.nn as nn
from torch.nn import ReLU
from torch.nn.parameter import Parameter

class LabelEncoder(nn.Module):
    def __init__(self, opt, hidden_size):
        super(LabelEncoder, self).__init__()
        model_path = opt['init_checkpoint']
        label_word_num = opt['label_word_num']

        if os.path.exists(model_path):
            state_dict = torch.load(model_path)
            for k, v in state_dict['state'].items():
                if k.find('word_embeddings') != -1:
                    self.word_embedding = nn.Embedding.from_pretrained(v)
                    break
        else:
            print('The label embedding will be initialized randomly!')
            self.word_embedding = nn.Embedding(opt['config']['vocab_size'], hidden_size)
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.activation = ReLU()
        # self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.average_label_word = nn.AvgPool1d(label_word_num, stride=label_word_num)

    def forward(self, label_word_ids):
        label_word = self.word_embedding(label_word_ids)
        label_word = self.activation(self.linear1(label_word))
        label_word = label_word.permute(0, 2, 1)
        label_embedding = self.average_label_word(label_word)
        label_embedding = label_embedding.permute(0, 2, 1)

        return label_embedding


class SeqMatchLabel(nn.Module):
    def __init__(self, opt, hidden_size):
        super(SeqMatchLabel, self).__init__()
        model_path = opt['init_checkpoint']
        label_word_num = opt['label_word_num']
        self.hidden_size = hidden_size
        if os.path.exists(model_path):
            state_dict = torch.load(model_path)
            for k, v in state_dict['state'].items():
                if k.find('word_embeddings') != -1:
                    self.word_embedding = nn.Embedding.from_pretrained(v)
                    break
        else:
            print('The label embedding will be initialized randomly!')
            self.word_embedding = nn.Embedding(opt['config']['vocab_size'], hidden_size)
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.activation = ReLU()
        self.average_label_word = nn.AvgPool1d(label_word_num, stride=label_word_num)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.seq_softmax = nn.Softmax(dim=1)
        self.label_softmax = nn.Softmax(dim=2)

    def forward(self, label_word_ids, seq, label_mask):

        label_word = self.word_embedding(label_word_ids)
        label_word = self.activation(self.linear1(label_word))
        label_word = label_word.permute(0, 2, 1)
        label = self.average_label_word(label_word)
        label = label.permute(0, 2, 1)

        seq = self.linear2(seq)
        label = self.linear3(label)

        similarity_matrix = torch.matmul(seq, label.permute(0, 2, 1)) / math.sqrt(
            self.hidden_size)  # size is [batch, seq_length, label_length]
        seq_weight = self.seq_softmax(similarity_matrix)
        label_weight = self.label_softmax(similarity_matrix)
        seq_to_label = torch.matmul(seq_weight.permute(0, 2, 1), seq)  # size is [batch, label_num, hidden_size]
        label_to_seq = torch.matmul(label_weight,
                                    torch.cat((label, seq_to_label), 2))  # size is [batch, seq_length, hidden_size * 2]

        if label_mask is not None:
            label_mask = label_mask.unsqueeze(2)
            seq_to_label = torch.matmul(seq_to_label.permute(0, 2, 1), label_mask)
            seq_to_label = seq_to_label.squeeze(2)
            assert seq_to_label.size() == (label.size(0), label.size(2))

        return label_to_seq, seq_to_label


class SeqMatchLabel2(nn.Module):
    def __init__(self, opt, hidden_size, lab):
        super(SeqMatchLabel2, self).__init__()
        self.model_path = opt['init_checkpoint']
        self.label_word_num = opt['label_word_num']
        self.lab = lab
        self.hidden_size = hidden_size
        if os.path.exists(self.model_path):
            state_dict = torch.load(self.model_path)
            for k, v in state_dict['state'].items():
                if k.find('word_embeddings') != -1:
                    self.word_embedding = nn.Embedding.from_pretrained(v)
                    break
        else:
            print('The label embedding will be initialized randomly!')
            self.word_embedding = nn.Embedding(opt['config']['vocab_size'], hidden_size)
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.activation = ReLU()
        self.average_label_word = nn.AvgPool1d(self.label_word_num, stride=self.label_word_num)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, hidden_size)
        self.seq_softmax = nn.Softmax(dim=3)
        self.label_softmax = nn.Softmax(dim=2)

    def forward(self, label_word_ids, seq, label_mask):

        label_word = self.word_embedding(label_word_ids)
        label_word = self.activation(self.linear1(label_word))

        # seq to label
        label_word = self.linear2(label_word)
        seq = self.linear3(seq)
        label_word = label_word.reshape(-1, self.lab, self.label_word_num, self.hidden_size)
        seq2label_matrix = torch.einsum('ijkl,iml->ijkm', [label_word, seq]) / math.sqrt(self.hidden_size)
        assert seq2label_matrix.size() == (seq.size(0), self.lab, self.label_word_num, seq.size(1))
        seq2label_weight = self.seq_softmax(seq2label_matrix)
        seq2label = torch.einsum('ijkm,iml->iljk', [seq2label_weight, seq])  # [batch, hidden_size, label_length, label_word_num]

        seq2label = self.average_label_word(seq2label.reshape(-1, self.hidden_size, self.lab * self.label_word_num)) # size is [batch, label_num, hidden_size]
        seq2label = seq2label.permute(0, 2, 1)

        assert seq2label.size() == (seq.size(0), self.lab, self.hidden_size)

        seq2label = self.linear4(seq2label)
        label2seq_matrix = torch.einsum('ijk,ilk->ijl', [seq, seq2label]) / math.sqrt(self.hidden_size)  # [batch, seq_length, label_length]
        label2seq_weight = self.label_softmax(label2seq_matrix)
        label2seq = torch.einsum('ijl,ilk->ijk', [label2seq_weight, seq2label])  # [batch, seq_length, hidden_size]

        if label_mask is not None:
            label_mask = label_mask.unsqueeze(2)
            seq2label = torch.matmul(seq2label.permute(0, 2, 1), label_mask)
            seq2label = seq2label.squeeze(2)
            assert seq2label.size() == (seq.size(0), self.hidden_size)

        return label2seq, seq2label


class SeqMatchLabel3(nn.Module):
    def __init__(self, opt, hidden_size, lab):
        super(SeqMatchLabel3, self).__init__()
        self.model_path = opt['init_checkpoint']
        self.label_word_num = opt['label_word_num']
        self.lab = lab
        self.hidden_size = hidden_size
        self.batch_size = opt['batch_size']
        if os.path.exists(self.model_path):
            state_dict = torch.load(self.model_path)
            for k, v in state_dict['state'].items():
                if k.find('word_embeddings') != -1:
                    self.word_embedding = nn.Embedding.from_pretrained(v)
                    break
        else:
            print('The label embedding will be initialized randomly!')
            self.word_embedding = nn.Embedding(opt['config']['vocab_size'], hidden_size)
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.activation = ReLU()
        self.average_label_word = nn.AvgPool1d(self.label_word_num, stride=self.label_word_num)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, hidden_size)
        self.bilstm = nn.LSTM(hidden_size*2, int(hidden_size / 2), batch_first=True, bidirectional=True)

        self.seq_softmax = nn.Softmax(dim=3)
        self.label_softmax = nn.Softmax(dim=2)

    def forward(self, label_word_ids, seq, label_mask):

        label_word = self.word_embedding(label_word_ids)
        label_word = self.activation(self.linear1(label_word))

        # seq to label
        label_word = self.linear2(label_word)
        sequence = self.linear3(seq)
        label_word = label_word.reshape(-1, self.lab, self.label_word_num, self.hidden_size)
        seq2label_matrix = torch.einsum('ijkl,iml->ijkm', [label_word, sequence]) / math.sqrt(self.hidden_size)
        assert seq2label_matrix.size() == (sequence.size(0), self.lab, self.label_word_num, sequence.size(1))
        seq2label_weight = self.seq_softmax(seq2label_matrix)
        seq2label = torch.einsum('ijkm,iml->iljk', [seq2label_weight, sequence])  # [batch, hidden_size, label_num, label_word_num]

        seq2label = self.average_label_word(seq2label.reshape(-1, self.hidden_size, self.lab * self.label_word_num)) # size is [batch, label_num, hidden_size]
        seq2label = seq2label.permute(0, 2, 1)

        assert seq2label.size() == (sequence.size(0), self.lab, self.hidden_size)

        seq2label = self.linear4(seq2label)
        label2seq_matrix = torch.einsum('ijk,ilk->ijl', [sequence, seq2label]) / math.sqrt(self.hidden_size)  # [batch, seq_length, label_num]
        label2seq_weight = self.label_softmax(label2seq_matrix)
        label2seq = torch.einsum('ijl,ilk->ijk', [label2seq_weight, seq2label])  # [batch, seq_length, hidden_size]
        concat_tensor = torch.cat((seq, label2seq), 2)
        label2seq, (h, c) = self.bilstm(concat_tensor)

        if label_mask is not None:
            label_mask = label_mask.unsqueeze(2)
            seq2label = torch.matmul(seq2label.permute(0, 2, 1), label_mask)
            seq2label = seq2label.squeeze(2)
            assert seq2label.size() == (sequence.size(0), self.hidden_size)
        return label2seq, seq2label


# if __name__ == '__main__':
#     label_word_id = torch.LongTensor([[2627, 4283, 11248, 2637, 4289, 3325, 2480, 597, 2467, 1397, 2629, 615, 3167, 508, 2718, 2627, 4283, 11248, 2637, 666, 2651, 4289, 3823, 3325, 2480, 597, 2999, 2467, 8678, 1397]])
#     opt = dict()
#     opt['init_checkpoint'] = '/data1/home/liuxiaofeng/code/bluebert-master/model/scibert_uncased/scibert.pt'
#     opt['config'] = {'vocab_size': 31090, 'hidden_size': 768, 'num_hidden_layers': 12, 'num_attention_heads': 12, 'hidden_act': 'gelu', 'intermediate_size': 3072, 'hidden_dropout_prob': 0.1, 'attention_probs_dropout_prob': 0.1, 'max_position_embeddings': 512, 'type_vocab_size': 2, 'initializer_range': 0.02}
#     label_encoder = LabelEncoder(opt, 768, 15)
#     result = label_encoder(label_word_id)
#     print(result)
#     seq = torch.FloatTensor(1, 3, 768).fill_(1)
#     print(seq.size())
#     match = SeqMatchLabel()
#     new_seq = match(seq, result)
