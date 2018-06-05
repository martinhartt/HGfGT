import torch
import torch.nn as nn
from util import apply_cuda
import math
from language_model import LanguageModel
from hier_attn import HierAttnDecoder, HierAttnEncoder
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
    parser.add_argument('--clip', type=float, default=0.25,
                        help='Gradient clipping')
    parser.add_argument(
        '--useTeacherForcing',
        type=bool,
        default=False,
        help="Use teacher forcing?")
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
        self.hier = opt.hier

        if opt.restore:
            if opt.hier:
                self.mlp, self.encoder = torch.load(opt.model)
            else:
                self.mlp = torch.load(opt.model)
                self.encoder = self.mlp.encoder

            self.mlp.epoch += 1
            print("Restoring MLP {} with epoch {}".format(
                opt.model, self.mlp.epoch))
        else:
            if opt.hier:
                glove_weights = build_glove(dict["w2i"]) if opt.glove else None

                self.encoder = apply_cuda(HierAttnEncoder(len(dict["i2w"]), opt.bowDim, opt.hiddenSize, opt, glove_weights))
                self.mlp = apply_cuda(HierAttnDecoder(len(dict["i2w"]), opt.bowDim, opt.hiddenSize, opt, glove_weights))
            else:
                self.mlp = apply_cuda(LanguageModel(self.dict, opt))
                self.encoder = self.mlp.encoder

            self.mlp.epoch = 0

        self.loss = apply_cuda(nn.NLLLoss(ignore_index=0))
        self.decoder_embedding = self.mlp.context_embedding

        if opt.hier:
            c = 0.9
            self.encoder_optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, self.encoder.parameters()), self.opt.learningRate,
                                                                    momentum=c, weight_decay=c)
            self.optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, self.mlp.parameters()), self.opt.learningRate,
                                                                    momentum=c, weight_decay=c)
        else:
            self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.mlp.parameters()), self.opt.learningRate)  # Half learning rate

    def validation(self, valid_data):
        offset = self.opt.batchSize
        loss = 0
        total = 0
        valid_data.reset()

        for (article, context), targets in valid_data.next_batch(offset):
            sample = (article, context), targets
            err = self.train_sample(sample)

            # Augment counters
            loss += float(err)
            total += int(targets.size(0))

        print("[perp: {} validation: {} total: {}]".format(
            math.exp(loss / total),
            loss / total,
            total
        ))
        return float(loss) / float(total)

    def run_valid(self, valid_data):
        # Run validation
        cur_valid_loss = self.validation(valid_data)
        # If valid loss does not improve drop learning rate
        if cur_valid_loss > self.last_valid_loss:
            if self.hier:
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

        if self.hier:
            if self.encoder.summary_embedding != None:
                self.renorm(self.encoder.summary_embedding.weight.data)
        else:
            if self.encoder.article_embedding != None:
                self.renorm(self.encoder.article_embedding.weight.data)

            if self.encoder.context_embedding != None:
                self.renorm(self.encoder.context_embedding.weight.data)

    def train_sample(self, sample):
        (article, context), targets = sample
        if self.hier:
            hidden_state = self.encoder.init_hidden()
            summ_hidden_state = self.encoder.init_hidden(n=self.opt.summLstmLayers, K=self.opt.K)
            encoder_out, _, _ = self.encoder(article, hidden_state, summ_hidden_state)

            err = 0

            teacher_forcing = self.opt.useTeacherForcing if random.random() < 0.5 else False
            if teacher_forcing:
                for i in range(len(targets)):
                    target = targets[i].unsqueeze(0)
                    ctx = context[i].unsqueeze(0)

                    out, hidden_state, _ = self.mlp(encoder_out, ctx, hidden_state)
                    err += self.loss(out, target)
            else:
                ctx = apply_cuda(torch.tensor(self.dict["w2i"]["<s>"]))
                for i in range(len(targets)):
                    target = targets[i].unsqueeze(0)
                    ctx = ctx.unsqueeze(0).unsqueeze(0)

                    out, hidden_state, _ = self.mlp(encoder_out, ctx, hidden_state)
                    err += self.loss(out, target)

                    topv, topi = out.topk(1)
                    ctx = topi.squeeze().detach()
        else:
            out, attn = self.mlp(article, context)
            err = self.loss(out, targets)

        return err

    def train(self, data, valid_data):
        print("Using cuda? {}".format(torch.cuda.is_available()))

        self.last_valid_loss = 1e9

        self.save()
        for epoch in range(self.mlp.epoch, self.opt.epochs):
            data.reset()
            # self.renorm_tables()
            self.run_valid(valid_data)
            self.mlp.epoch = epoch

            # Loss for the epoch
            epoch_loss = 0
            batch = 0
            last_batch = 0
            total = 0
            loss = 0

            for sample in data.next_batch(self.opt.batchSize):
                self.optimizer.zero_grad()
                if self.hier:
                    self.encoder_optimizer.zero_grad()

                err = self.train_sample(sample)

                err.backward()

                if self.hier:
                    torch.nn.utils.clip_grad_norm(filter(lambda p: p.requires_grad, self.mlp.parameters()), self.opt.clip)
                    torch.nn.utils.clip_grad_norm(filter(lambda p: p.requires_grad, self.encoder.parameters()), self.opt.clip)

                self.optimizer.step()
                if self.hier:
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

        state = (self.mlp, self.encoder) if self.hier else self.mlp

        torch.save(state, self.opt.model)
        print('Finished saving')
        # Save current epoch for evaluation purposes
        # if self.mlp.epoch is not None:
        #     torch.save(state, "{}__{}".format(self.opt.model, self.mlp.epoch))
