#coding:utf-8

# import modules related to torch
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import init
from torchqrnn import QRNN

import os
import time

# set parameters
HIDDEN_SIZE =
N_LAYER =
N_EPOCHS =

N_CHARS =
N_LABEL =
USE_GPU =

class QRNNClassifier(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1, bidirectional=True):
        super(QRNNClassifier, self).__init__()

        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_directions = 2 if bidirectional else 1

        self.embedding = nn.Embedding(input_size, hidden_size)

        self.gru = QRNN(hidden_size, hidden_size, n_layers, dropout=0.4)

        self.fc = nn.Linear(hidden_size * self.n_directions, output_size)

    def _init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers * self.n_directions, batch_size, self.hidden_size)
        return create_tensor(hidden)

    def forward(self, input):
        # (batchSize,seqLen) -> (seqLen,batchSize)
        input = input.t()

        # save batch_size for making initial hidden
        batch_size = input.size(1)

        hidden = self._init_hidden(batch_size)

        embedding = self.embedding(input)

        # QRNN
        if USE_GPU:
            self.gru.cuda()
        output, hidden = self.gru(embedding, hidden)

        if self.n_directions == 2:
            # if we use bidirectional GRU, the forward hidden and backward hidden should be concatenate
            hidden_cat = torch.cat([hidden[-1], hidden[-2]], dim=1)
        else:
            hidden_cat = hidden[-1]

        # use linear classifier
        fc_output = self.fc(hidden_cat)
        return fc_output

def message2list(dataset_dir, file, label):
    file_dir = os.path.join(dataset_dir, file)
    if os.path.isfile(file_dir):
        # read by bytes
        with open(file_dir, "rb") as f:
            arr = [byt+1 for byt in f.read()]

        if label == "trainset":
            # add your label
            if "label" in dataset_dir:
                return arr, len(arr), 0
            else:
                return arr, len(arr), 1
        elif label == "testset":
            # add your label
            if "label" in dataset_dir:
                return arr, len(arr), 0
            else:
                return arr, len(arr), 1
        else:
            print("Neither trainset nor testset")
            exit(0)

def create_tensor(tensor):
    if USE_GPU:
        device = torch.device("cuda:0")
        tensor = tensor.to(device)
    return tensor

def make_tensors(dataset, label):
    sequences_and_lengths = []
    for dataset_dir in dataset:
        sequences_and_lengths += [message2list(dataset_dir, file, label) for file in os.listdir(dataset_dir)]

    sequences_and_lengths = list(filter(None, sequences_and_lengths))
    if label == "trainset":
        print("Number of training sequences: ", len(sequences_and_lengths))
    else:
        print("Number of testing sequences: ", len(sequences_and_lengths))

    message_sequences = [ml[0] for ml in sequences_and_lengths]
    seq_lengths = torch.LongTensor([ml[1] for ml in sequences_and_lengths])
    labels = torch.LongTensor([ml[2] for ml in sequences_and_lengths])

    # make tensor of message, padding, BatchSize * SeqLen
    seq_tensor = torch.zeros(len(message_sequences), seq_lengths.max()).long()
    for idx, (seq, seq_len) in enumerate(zip(message_sequences, seq_lengths), 0):
        seq_tensor[idx, :seq_len] = torch.LongTensor(seq)

    return create_tensor(seq_tensor),\
           create_tensor(seq_lengths),\
           create_tensor(labels)

def trainModel(epoch,trainset):
    total_loss = 0
    inputs, seq_lengths, target = make_tensors(trainset, "trainset")
    output = classifier(inputs)
    loss = criterion(output, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    total_loss += loss.item()
    print("Epoch[{}/{}], Loss: {:.5f}".format(epoch, N_EPOCHS, total_loss))

def testModel(testset):
    correct = 0
    print("Evaluating trained model ...")
    with torch.no_grad():
        inputs, seq_lengths, target = make_tensors(testset, "testset")
        output = classifier(inputs)
        pred = output.max(dim=1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()
        total = len(inputs)
        percent = "%.2f" % (100 * float(correct) / total)
        print("Testset: Accuracy {}/{} {}%".format(correct, total, percent))

if __name__ == "__main__":
    # instantiate the classifier model
    classifier = QRNNClassifier(N_CHARS, HIDDEN_SIZE, N_LABEL, N_LAYER)
    # whether use GPU for training model
    if USE_GPU:
        device = torch.device("cuda:0")
        classifier.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=0.001)

    start_time = time.time()
    print("Training for %d epochs..." % N_EPOCHS)
    for epoch in range(1, N_EPOCHS + 1):
        # add your trainset
        trainset = []
        trainModel(epoch,trainset)
    end_time = time.time()
    print("The model has been trained for {:.2f}hrs\n".format((end_time - start_time)/60/60))

    # add your testset
    testset = []
    testModel(testset)

    # save your model
    torch.save(classifier.state_dict(), "your_model_name.pkl")
