import torch
import torch.nn as nn
from util import apply_cuda
import math
from language_model import LanguageModel


def addOpts(parser):
    parser.add_argument(
        '-epochs', type=int, default=15, help="Number of epochs to train.")
    parser.add_argument(
        '-printEvery',
        type=int,
        default=10000,
        help="How often to print during training.")
    parser.add_argument(
        '-modelFilename', default='', help="File for saving loading/model.")
    parser.add_argument(
        '-window', type=int, default=5, help="Size of NNLM window.")
    parser.add_argument(
        '-embeddingDim', type=int, default=50, help="Size of NNLM embeddings.")
    parser.add_argument(
        '-hiddenSize',
        type=int,
        default=100,
        help="Size of NNLM hiddent layer.")
    parser.add_argument(
        '-learningRate', type=float, default=0.1, help="SGD learning rate.")
    parser.add_argument(
        '-restore',
        type=bool,
        default=False,
        help="Should a previous model be restored?")


class NNLM(object):
    """docstring for NNLM."""

    def __init__(self, opt, dictionary, encoder, encoder_size):
        super(NNLM, self).__init__()
        self.opt = opt
        self.dict = dictionary
        self.encoder = encoder

        if opt.restore:
            self.mlp = torch.load(opt.modelFilename)
            self.mlp.epoch += 1
            print("Restoring MLP {} with epoch {}".format(
                opt.modelFilename, self.mlp.epoch))
        else:
            self.mlp = apply_cuda(
                LanguageModel(encoder, encoder_size, self.dict, opt))
            self.mlp.epoch = 0

        self.loss = nn.NLLLoss()
        self.embedding = self.mlp.context_embedding
        self.optimizer = torch.optim.SGD(
            self.mlp.parameters(), self.opt.learningRate)  # Half learning rate

    def validation(self, valid_data):
        offset = self.opt.miniBatchSize
        loss = 0
        total = 0
        valid_data.reset()

        while not valid_data.is_done():
            input, target = valid_data.next_batch(offset)
            out = self.mlp(*input)
            err = self.loss(out, target) * target.size(0)

            # Augment counters
            loss += float(err)
            total += int(target.size(0))

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
                self.opt.learningRate = self.opt.learningRate / 2

                for param_group in optimizer.param_groups:
                    param_group['lr'] = self.opt.learningRate

            self.last_valid_loss = cur_valid_loss

    def renorm(self, data, th=1):
        size = data.size(0)
        for i in range(size):
            norm = float(data[i].norm())
            if norm > th:
                data[i] = data[i].div(norm / th)

    def renorm_tables(self):
        if self.embedding != None:
            self.renorm(self.embedding.weight.data)

        if self.encoder.article_embedding != None:
            self.renorm(self.encoder.article_embedding.weight.data)

        if self.encoder.title_embedding != None:
            self.renorm(self.encoder.title_embedding.weight.data)

    def train(self, data, valid_data):
        print("Using cuda? {}".format(torch.cuda.is_available()))

        self.last_valid_loss = 1e9

        for epoch in range(self.mlp.epoch, self.opt.epochs):
            data.reset()
            self.renorm_tables()
            self.run_valid(valid_data)
            self.mlp.epoch = epoch

            # Loss for the epoch
            # epoch_loss = 0
            # batch = 0
            # last_batch = 0
            # total = 0
            # loss = 0

            for batch in data.next_batch():
                self.optimizer.zero_grad()

                losses = []
                for input, target in batch:
                    out = self.mlp(input)
                    err = self.loss(out, target)

                err.backward()
                self.optimizer.step()

                loss += float(err)
                epoch_loss += float(err)

                if (batch % self.opt.printEvery) == 0:
                    print(
                        "[Loss: {} Loss2: {} Epoch: {} Position: {} Rate: {}]".
                        format(loss / ((
                            batch - last_batch + 1) * self.opt.miniBatchSize),
                               loss /
                               (self.opt.printEvery * self.opt.miniBatchSize),
                               epoch, batch * self.opt.miniBatchSize,
                               self.opt.learningRate))
                    last_batch = batch
                    loss = 0

                batch += 1
                total += input[0].data.size(0)

            self.save()
            print("[EPOCH : {} LOSS: {} TOTAL: {} BATCHES: {}]".format(
                epoch, epoch_loss / total, total, batch))
            exit(1)

    def save(self):
        print('Saving...')
        torch.save(self.mlp, self.opt.modelFilename)
        # Save current epoch for evaluation purposes
        if self.mlp.epoch is not None:
            torch.save(self.mlp, "{}__{}".format(self.opt.modelFilename,
                                                 self.mlp.epoch))
