import data
import features

INF = 1e9

def addOpts(parser):
    parser.add_argument('-allowUNK', type=bool,          default=False, help="Allow generating <unk>.")
    parser.add_argument('-fixedLength', type=bool,       default=True,  help="Produce exactly -length words.")
    parser.add_argument('-blockRepeatWords', type=bool,  default=False, help="Disallow generating a word twice.")
    parser.add_argument('-lmWeight', type=float,            default=1.0, help="Weight for main model.")
    parser.add_argument('-beamSize', type=int,            default=100, help="Size of the beam.")
    parser.add_argument('-extractive', type=bool,        default=False, help="Force fully extractive summary.")
    parser.add_argument('-abstractive', type=bool,       default=False, help="Force fully abstractive summary.")
    parser.add_argument('-recombine', type=bool,         default=False, help="Used hypothesis recombination.")
    features.addOpts(parser)

class Beam(object):
    """docstring for Beam."""
    def __init__(self, mlp, aux_model, article_to_title, dict):
        super(Beam, self).__init__()

        self.opt = opt
        self.K = opt.beamSize
        self.mlp = mlp
        self.aux_model = aux_model
        self.article_to_title = article_to_title
        self.dict = dict

        # Special Symbols
        self.UNK = dict.symbol_to_index["<unk>"]
        self.START = dict.symbol_to_index["<s>"]
        self.END = dict.symbol_to_index["</s>"]

    # Use beam search to generate a summary of the article of length < length
    def generate(self, article, length):
        n = length
        K = self.K
        W = self.opt.window

        # Initialise the extractive features
        feat_gen = features.Features(self.opt, self.article_to_title)
        feat_gen.match_words(self.START, article)
