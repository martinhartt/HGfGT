import torch
import torch.nn as nn
from util import apply_cuda
import math
from language_model import LanguageModel
from heir_attn import HeirAttnDecoder, HeirAttnEncoder
from glove import build_glove
from torch.autograd import Variable
import random

def add_opts(parser):
    parser.add_argument(
        '--epochs', type=int, default=100, help="Number of epochs to train.")
    parser.add_argument(
        '--printEvery',
        type=int,
        default=10000,
        help="How often to print during training.")
    parser.add_argument(
        '--model', default='', help="File for saving loading/model.")
    parser.add_argument(
        '--window', type=int, default=5, help="Size of Trainer window.")
    parser.add_argument(
        '--hiddenSize',
        type=int,
        default=512,
        help="Size of hidden layer.")
    parser.add_argument(
        '--attentionDims',
        type=int,
        default=40,
        help="Num of dimensions for attention.")
    parser.add_argument(
        '--learningRate', type=float, default=0.1, help="SGD learning rate.")
    parser.add_argument(
        '--restore',
        type=bool,
        default=False,
        help="Should a previous model be restored?")
    parser.add_argument(
        '--batchSize',
        type=int,
        default=64,
        help="Size of training minibatch.")


class Trainer(object):
    """docstring for Trainer."""

    def __init__(self, opt, dict):
        super(Trainer, self).__init__()
        self.opt = opt
        self.dict = dict
        self.heir = opt.heir

        if opt.restore:
            if opt.heir:
                self.mlp, self.encoder = torch.load(opt.model)
            else:
                self.mlp = torch.load(opt.model)

            self.mlp.epoch += 1
            print("Restoring MLP {} with epoch {}".format(
                opt.model, self.mlp.epoch))
        else:
            if opt.heir:
                glove_weights = build_glove(dict["w2i"]) if opt.glove else None

                self.encoder = apply_cuda(HeirAttnEncoder(len(dict["i2w"]), opt.bowDim, opt.hiddenSize, opt, glove_weights))
                self.mlp = apply_cuda(HeirAttnDecoder(len(dict["i2w"]), opt.bowDim, opt.hiddenSize, opt, glove_weights))
            else:
                self.mlp = apply_cuda(LanguageModel(self.dict, opt))
                self.encoder = self.mlp.encoder

            self.mlp.epoch = 0

        self.loss = apply_cuda(nn.NLLLoss(ignore_index=0))
        self.decoder_embedding = self.mlp.context_embedding

        if opt.heir:
            c = 0.9
            self.encoder_optimizer = torch.optim.RMSprop(self.encoder.parameters(), self.opt.learningRate,
                                                                    momentum=c, weight_decay=c)
            self.optimizer = torch.optim.RMSprop(self.mlp.parameters(), self.opt.learningRate,
                                                                    momentum=c, weight_decay=c)
        else:
            self.optimizer = torch.optim.SGD(self.mlp.parameters(), self.opt.learningRate)  # Half learning rate

    def validation(self, valid_data):
        offset = self.opt.batchSize
        loss = 0
        total = 0
        valid_data.reset()

        for (article, context), targets in valid_data.next_batch(offset):
            if self.heir:
                encoder_out = self.encoder(article)
                out = self.mlp(encoder_out, context)
            else:
                out = self.mlp(article, context)

            err = self.loss(out, targets) * targets.size(0)

            # Augment counters
            loss += float(err)
            total += int(targets.size(0))

        print("[perp: %f validation: %f total: %d]".format(
            math.exp(loss / total),
            loss / total,
        ))
        return float(loss) / float(total)

    def run_valid(self, valid_data):
        # Run validation
        if valid_data != None:
            cur_valid_loss = self.validation(valid_data)
            # If valid loss does not improve drop learning rate
            if cur_valid_loss > self.last_valid_loss:
                if self.heir:
                    if self.mlp.epoch > 20:
                        self.save()
                        print("Loss is no longer decreasing for validation - stopping training...")
                        exit(1)
                else:
                    self.opt.learningRate = self.opt.learningRate / 2

                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.opt.learningRate

            self.last_valid_loss = cur_valid_loss

    def renorm(self, data, th=1):
        size = data.size(0)
        for i in range(size):
            norm = float(data[i].norm())
            if norm > th:
                data[i] = data[i].div(norm / th)

    def renorm_tables(self):
        if self.decoder_embedding != None:
            self.renorm(self.decoder_embedding.weight.data)

        if self.heir:
            if self.encoder.summary_embedding != None:
                self.renorm(self.encoder.summary_embedding.weight.data)
        else:
            if self.encoder.article_embedding != None:
                self.renorm(self.encoder.article_embedding.weight.data)

            if self.encoder.context_embedding != None:
                self.renorm(self.encoder.context_embedding.weight.data)



    def train(self, data, valid_data):
        print("Using cuda? {}".format(torch.cuda.is_available()))

        self.last_valid_loss = 1e9

        self.save()
        for epoch in range(self.mlp.epoch, self.opt.epochs):
            data.reset()
            self.renorm_tables()
            # self.run_valid(valid_data)
            self.mlp.epoch = epoch

            # Loss for the epoch
            epoch_loss = 0
            batch = 0
            last_batch = 0
            total = 0
            loss = 0

            for (article, context), targets in data.next_batch(self.opt.batchSize):
                self.optimizer.zero_grad()
                if self.heir:
                    self.encoder_optimizer.zero_grad()

                    encoder_out = self.encoder(article)

                    err = 0
                    for i in range(len(targets)):
                        target = targets[i].unsqueeze(0)
                        assert i+1 == context[1][i]
                        ctx = context[0][i][:i+1].unsqueeze(0), [context[1][i]]

                        out = self.mlp(encoder_out, ctx)
                        err += self.loss(out, target)
                else:
                    out = self.mlp(article, context)
                    err = self.loss(out, targets)


                err.backward()
                self.optimizer.step()
                if self.heir:
                    self.encoder_optimizer.step()

                loss += float(err)
                epoch_loss += float(err)

                if (batch % self.opt.printEvery) == 0:
                    print(
                        "[Loss: {} Epoch: {} Position: {} Rate: {}]".format(
                            loss / (batch - last_batch + 1),
                            epoch,
                            batch * self.opt.batchSize,
                            self.opt.learningRate))
                    last_batch = batch
                    loss = 0

                batch += 1
                total += 1

            self.save()
            print("[EPOCH : {} LOSS: {} TOTAL: {} BATCHES: {}]".format(
                epoch, epoch_loss / total, total, batch))

    def save(self):
        print('Saving...')

        state = (self.mlp, self.encoder) if self.heir else self.mlp

        torch.save(state, self.opt.model)
        print('Finished saving')
        # Save current epoch for evaluation purposes
        # if self.mlp.epoch is not None:
        #     torch.save(state, "{}__{}".format(self.opt.model, self.mlp.epoch))
