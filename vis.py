#!/usr/bin/env python
# -*- coding: utf-8 -*-
# !@Time    : 2021/5/26 上午10:51
# !@Author  : miracleyin @email: miracleyin@live.com
# !@File    : vis.py
import torch
import math
import torch
import torch.nn as nn
import time
from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer

from collections import Counter
from torchtext.vocab import Vocab
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class Corpus(object):
    def __init__(self, train_batch_size=20, eval_batch_size=10, pred_batch_size=1, bptt=35):
        # 'train': 36718, 'valid': 3760, 'test': 4358,

        self.bptt = bptt
        train_iter = WikiText2(split='train')
        self.tokenizer = get_tokenizer('basic_english')
        counter = Counter()
        txtline = []
        for line in train_iter:
            txtline.append(line)
            counter.update(self.tokenizer(line))
        self.vocab = Vocab(counter)
        train_iter, val_iter, test_iter = WikiText2()

        train_data = self.data_process(train_iter)
        val_data = self.data_process(val_iter)
        test_data = self.data_process(test_iter)
        pred_data = train_data

        self.train_data = self.batchify(train_data, train_batch_size)
        self.val_data = self.batchify(val_data, eval_batch_size)
        self.test_data = self.batchify(test_data, eval_batch_size)
        self.pred_data = self.batchify(pred_data, pred_batch_size)  # 用于单行预测
        self.text = txtline

    def data_process(self, raw_text_iter):
        data = [torch.tensor([self.vocab[token] for token in self.tokenizer(item)],
                             dtype=torch.long) for item in raw_text_iter] # 这里的data长度和对应数据集长度相匹配
        a = filter(lambda t: t.numel() > 0, data) # 把长度为0的行删掉
        b = tuple(a)# 转化为元组
        c = torch.cat(b)# 直接拼成一行
        return c
        return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

    def single_data_process(self, single_text):
        data = torch.tensor([self.vocab[token] for token in single_text], dtype=torch.long)
        return data

    def batchify(self, data, batch_size):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Divide the dataset into batch_size parts.
        nbatch = data.size(0) // batch_size
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data.narrow(0, 0, nbatch * batch_size)
        # Evenly divide the data across the batch_size batches.
        data = data.view(batch_size, -1).t().contiguous()
        return data.to(device)

    def get_batch(self, source, i):
        seq_len = min(self.bptt, len(source) - 1 - i)
        data = source[i:i + seq_len]
        target = source[i + 1:i + 1 + seq_len].reshape(-1)
        return data, target

    def get_ntokens(self):
        return len(self.vocab.stoi)

    def get_text(self, raw_text_iter):
        pass


class TransformerModel(nn.Module):

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()

        ########################################
        ######Your code here########
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)
        ########################################
        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        ########################################
        ######Your code here########
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
        ########################################
        # pass

    def init_weights(self):
        initrange = 0.1
        encoder_weight = self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        decoder_weight = self.decoder.weight.data.uniform_(-initrange, initrange)
        return encoder_weight, decoder_weight

    def forward(self, src, src_mask):
        ########################################
        ######Your code here########
        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output_transformer = self.transformer_encoder(src, src_mask)
        output = self.decoder(output_transformer)

        return F.log_softmax(output, dim=-1), output_transformer
        ########################################
        # pass


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        ########################################
        ######Your code here########
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (- math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        ########################################
        # pass

    def forward(self, x):
        ########################################
        ######Your code here########
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
        ########################################
        # pass


class Argument(object):
    def __init__(self):
        self.epochs = 1
        self.train_batch_size = 20
        self.eval_batch_size = 10
        self.pred_batch_size = 1
        self.bptt = 35
        self.seed = 1234
        self.is_train = False


def train():
    model.train()  # Turn on the train mode
    total_loss = 0.
    start_time = time.time()
    src_mask = model.generate_square_subsequent_mask(args.bptt).to(device)

    for batch, i in enumerate(range(0, data_loader.train_data.size(0) - 1, args.bptt)):
        data, targets = data_loader.get_batch(data_loader.train_data, i)
        optimizer.zero_grad()

        if data.size(0) != args.bptt:
            src_mask = model.generate_square_subsequent_mask(data.size(0)).to(device)

        output = model(data, src_mask)
        loss = criterion(output.view(-1, ntoken), targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        log_interval = 200

        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(data_loader.train_data) // args.bptt,
                scheduler.get_last_lr()[0],
                              elapsed * 1000 / log_interval,
                cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

    return math.exp(cur_loss)


# WRITE CODE HERE within two '#' bar
########################################
# Evaluation Function
# Calculate the average cross-entropy loss between the prediction and the ground truth word.
# And then exp(average cross-entropy loss) is perplexity.

def evaluate(eval_model, data_source):
    """

    :param eval_model:
    :param data_source:  size(-1, batch_size)
    :return:
    """
    eval_model.eval()  # Turn on the evaluation mode
    total_loss = 0.
    src_mask = model.generate_square_subsequent_mask(args.bptt).to(device)
    attn_weight = []
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets = data_loader.get_batch(data_source, i)

            ########################################
            ######Your code here########
            ########################################
            if data.size(0) != args.bptt:
                src_mask = model.generate_square_subsequent_mask(data.size(0)).to(device)

            output, output_transformer = eval_model(data, src_mask)
            attn_weight.append(output_transformer)
            output_flat = output.view(-1, ntoken)
            total_loss += len(data) * criterion(output_flat, targets).item()
    attn_weight = torch.cat(attn_weight, 0)
    attn_weight = torch.squeeze(attn_weight, dim=1)

    attn_mat = torch.dist(attn_weight, attn_weight)

    return total_loss / (len(data_source) - 1), attn_weight, attn_mat





if __name__ == '__main__':
    args = Argument()

    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)

    # load data
    data_loader = Corpus(train_batch_size=args.train_batch_size,
                         eval_batch_size=args.eval_batch_size,
                         pred_batch_size=args.pred_batch_size,
                         bptt=args.bptt)

    # WRITE CODE HERE within two '#' bar
    ########################################
    # bulid your language model here
    ntoken = data_loader.get_ntokens()
    model = TransformerModel(ntoken=data_loader.get_ntokens(), ninp=200, nhid=200, nhead=2, nlayers=2, dropout=0.2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    ########################################
    criterion = nn.CrossEntropyLoss()
    lr = 5.0  # learning rate
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
    if args.is_train:

        # Train Function
        best_val_loss = float("inf")
        epochs = 3,
        best_model = None

        train_loss_list = []
        val_loss_list = []

        for epoch in range(1, args.epochs + 1):
            epoch_start_time = time.time()
            train_loss = train()
            train_loss_list.append(train_loss)

            val_loss = evaluate(model, data_loader.val_data)
            val_loss_list.append(val_loss)

            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                  'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                             val_loss, math.exp(val_loss)))
            print('-' * 89)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = model
                torch.save(best_model, "./best.pt")

                # attn_weight = best_model.transformer_encoder.layers[0].self_attn

            scheduler.step()

    ########################################
    else:  # 测试模式
        model = torch.load("./best.pt")
        # test_loss = evaluate(model, data_loader.test_data)
        # print('=' * 89)
        # print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        #    test_loss, math.exp(test_loss)))
        # print('=' * 89)
        import time
        text = data_loader.text

        _, attn_weight, attn_mat = evaluate(model, data_loader.pred_data)

        print(attn_weight)
