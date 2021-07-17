import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from module.common import bertgelu
from torch.nn import ReLU


class BiaffinePairwiseScore(nn.Module):
    def __init__(self, hidden_size, label_num):
        super(BiaffinePairwiseScore, self).__init__()
        # self.conv1 = nn.Conv1d(hidden_size, 200, 3, padding=1)
        # self.conv2 = nn.Conv1d(hidden_size, 200, 3, padding=1)
        # self.conv3 = nn.Conv1d(200, 200, 5, padding=2)
        # self.conv4 = nn.Conv1d(200, 200, 5, padding=2)

        self.linear1 = nn.Linear(hidden_size, 200)
        self.linear2 = nn.Linear(hidden_size, 200)
        self.linear3 = nn.Linear(200, 200)
        self.linear4 = nn.Linear(200, 200)

        self.label_num = label_num
        self.hidden_size = hidden_size
        self.seq_length = 0
        self.activation = ReLU()
        self.weight = Parameter(torch.FloatTensor(200, label_num * 200))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight)

    def forward(self, seq, ep_dist):
        assert seq.ndim == 3
        seq_permute = seq.permute(0, 2, 1)
        self.seq_length = seq.shape[1]
        # head = bertgelu(self.conv3(bertgelu(self.conv1(seq_permute))))
        # tail = bertgelu(self.conv4(bertgelu(self.conv2(seq_permute))))

        # head = self.conv3(self.activation(self.conv1(seq_permute)))
        # tail = self.conv4(self.activation(self.conv2(seq_permute)))
        # head = head.permute(0, 2, 1)

        head = self.linear3(self.activation(self.linear1(seq)))
        tail = self.linear4(self.activation(self.linear2(seq)))
        tail = tail.permute(0, 2, 1)
        
        affine = torch.matmul(head, self.weight)  # size is (batch, seq_length, label_num, 200)

        bi_affine = torch.matmul(torch.reshape(affine, (-1, self.seq_length * self.label_num, 200)), tail)
        bi_affine = torch.reshape(bi_affine, (-1, self.seq_length, self.label_num, self.seq_length))
        bi_affine = bi_affine.permute(0, 1, 3, 2)

        ep_dist = ep_dist.unsqueeze(3)
        result = bi_affine + ep_dist
        output = torch.logsumexp(result, (1, 2))

        return output, result


class BiaffinePairwiseScore_2(nn.Module):
    def __init__(self, hidden_size, label_num):
        super(BiaffinePairwiseScore_2, self).__init__()
        self.conv1 = nn.Conv1d(hidden_size, 200, 3, padding=1)
        self.conv2 = nn.Conv1d(hidden_size, 200, 3, padding=1)
        self.conv3 = nn.Conv1d(200, 200, 3, padding=1)
        self.conv4 = nn.Conv1d(200, 200, 3, padding=1)

        # self.linear1 = nn.Linear(hidden_size, 200)
        # self.linear2 = nn.Linear(hidden_size, 200)
        # self.linear3 = nn.Linear(200, 200)
        # self.linear4 = nn.Linear(200, 200)

        self.linear = nn.Linear(hidden_size, label_num)
        self.label_num = label_num
        self.hidden_size = hidden_size
        self.seq_length = 0
        self.weight = Parameter(torch.FloatTensor(200, label_num * 200))
        self.activation = ReLU()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight)

    def forward(self, seq, cls_token, ep_dist):
        assert seq.ndim == 3
        seq_permute = seq.permute(0, 2, 1)
        self.seq_length = seq.shape[1]
        # head = bertgelu(self.conv3(bertgelu(self.conv1(seq_permute))))
        # tail = bertgelu(self.conv4(bertgelu(self.conv2(seq_permute))))

        head = self.conv3(self.activation(self.conv1(seq_permute)))
        tail = self.conv4(self.activation(self.conv2(seq_permute)))
        head = head.permute(0, 2, 1)
        #
        # head = self.linear3(self.activation(self.linear1(seq)))
        # tail = self.linear4(self.activation(self.linear2(seq)))
        # tail = tail.permute(0, 2, 1)

        affine = torch.matmul(head, self.weight)  # size is (batch, seq_length, label_num, 200)
        bi_affine = torch.matmul(torch.reshape(affine, (-1, self.seq_length * self.label_num, 200)), tail)
        bi_affine = torch.reshape(bi_affine, (-1, self.seq_length, self.label_num, self.seq_length))
        bi_affine = bi_affine.permute(0, 1, 3, 2)

        ep_dist = ep_dist.unsqueeze(3)
        local_matrix = bi_affine + ep_dist
        local_result = torch.logsumexp(local_matrix, (1, 2))

        global_result = self.linear(cls_token)

        output = local_result + global_result

        return output, local_matrix


class BiaffinePairwiseScore_3(nn.Module):
    def __init__(self, hidden_size, label_num):
        super(BiaffinePairwiseScore_3, self).__init__()
        self.conv1 = nn.Conv1d(hidden_size, 200, 7, padding=3)
        self.conv2 = nn.Conv1d(hidden_size, 200, 7, padding=3)
        self.conv3 = nn.Conv1d(200, 200, 7, padding=3)
        self.conv4 = nn.Conv1d(200, 200, 7, padding=3)

        # self.linear1 = nn.Linear(hidden_size, 200)
        # self.linear2 = nn.Linear(hidden_size, 200)
        # self.linear3 = nn.Linear(200, 200)
        # self.linear4 = nn.Linear(200, 200)

        self.linear = nn.Linear(hidden_size, label_num)
        self.bilstm = nn.LSTM(hidden_size, int(hidden_size / 2), batch_first=True, bidirectional=True)
        # self.conv1d = nn.Conv1d(hidden_size, hidden_size, 5, padding=2)
        # self.pooling = nn.MaxPool1d(512)

        self.label_num = label_num
        self.hidden_size = hidden_size
        self.seq_length = 0
        self.weight = Parameter(torch.FloatTensor(200, label_num * 200))
        self.activation = ReLU()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight)

    def forward(self, seq, ep_dist):
        assert seq.ndim == 3
        seq_permute = seq.permute(0, 2, 1)
        self.seq_length = seq.shape[1]

        head = self.conv3(self.activation(self.conv1(seq_permute)))
        tail = self.conv4(self.activation(self.conv2(seq_permute)))
        head = head.permute(0, 2, 1)

        # head = self.linear3(self.activation(self.linear1(seq)))
        # tail = self.linear4(self.activation(self.linear2(seq)))
        # tail = tail.permute(0, 2, 1)

        affine = torch.matmul(head, self.weight)  # size is (batch, seq_length, label_num, 200)
        bi_affine = torch.matmul(torch.reshape(affine, (-1, self.seq_length * self.label_num, 200)), tail)
        bi_affine = torch.reshape(bi_affine, (-1, self.seq_length, self.label_num, self.seq_length))
        bi_affine = bi_affine.permute(0, 1, 3, 2)

        ep_dist = ep_dist.unsqueeze(3)
        local_matrix = bi_affine + ep_dist
        local_result = torch.logsumexp(local_matrix, (1, 2))

        global_sequence, (h, c) = self.bilstm(seq)
        global_result = self.linear(global_sequence[:, 0, :])
        # hidden_output = self.pooling(self.conv1d(seq_permute))
        # hidden_output = hidden_output.squeeze(2)
        # global_result = self.linear(hidden_output)

        # print(self.linear.weight[0, 0])
        output = local_result + global_result

        return output, local_matrix


class RE(nn.Module):
    def __init__(self, hidden_size, lab):
        super(RE, self).__init__()
        self.bilstm = nn.LSTM(hidden_size, int(hidden_size / 2), batch_first=True, bidirectional=True)
        # self.conv1d = nn.Conv1d(hidden_size, hidden_size, 5, padding=3)
        # self.pooling = nn.MaxPool1d(512)
        self.linear = nn.Linear(hidden_size, lab)

    def forward(self, seq):
        hidden_output, (h, c) = self.bilstm(seq)
        output = self.linear(hidden_output[:, 0, :])
        # seq_permute = seq.permute(0, 2, 1)
        # hidden_output = self.pooling(self.conv1d(seq_permute))
        # hidden_output = hidden_output.squeeze(2)
        # output = self.linear(hidden_output)
        return output

