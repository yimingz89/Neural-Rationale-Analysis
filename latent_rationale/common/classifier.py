from torch import nn
import numpy as np

from latent_rationale.common.util import get_encoder


class Classifier(nn.Module):
    """
    The Encoder takes an input text (and rationale z) and computes p(y|x,z)

    Supports a sigmoid on the final result (for regression)
    If not sigmoid, will assume cross-entropy loss (for classification)

    """

    def __init__(self,
                 embed:        nn.Embedding = None,
                 hidden_size:  int = 200,
                 output_size:  int = 1,
                 dropout:      float = 0.1,
                 layer:        str = "rcnn",
                 nonlinearity: str = "sigmoid"
                 ):

        super(Classifier, self).__init__()

        emb_size = embed.weight.shape[1]

        self.embed_layer = nn.Sequential(
            embed,
            nn.Dropout(p=dropout)
        )

        self.enc_layer = get_encoder(layer, emb_size, hidden_size)

        if hasattr(self.enc_layer, "cnn"):
            enc_size = self.enc_layer.cnn.out_channels
        else:
            enc_size = hidden_size * 2

        self.output_layer = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(enc_size, output_size),
            nn.Sigmoid() if nonlinearity == "sigmoid" else nn.LogSoftmax(dim=-1)
        )

        self.report_params()

    def report_params(self):
        # This has 1604 fewer params compared to the original, since only 1
        # aspect is trained, not all. The original code has 5 output classes,
        # instead of 1, and then only supervise 1 output class.
        count = 0
        for name, p in self.named_parameters():
            if p.requires_grad and "embed" not in name:
                count += np.prod(list(p.shape))
        print("{} #params: {}".format(self.__class__.__name__, count))

    def forward(self, x, mask, z=None):

        rnn_mask = mask
        emb = self.embed_layer(x)

        # apply z to main inputs
        if z is not None:
            z_mask = (mask.float() * z).unsqueeze(-1)  # [B, T, 1] - note that an entry of mask.float() is 1 -> nonzero score, 0 -> zero score, also the * z just weights it by the original score (not necessarily 1)
            #print(z_mask)
            rnn_mask = z_mask.squeeze(-1) > 0.  # z could be continuous
            emb = emb * z_mask # of the 300 25x34 embeddings (300 = dimension of embeddings, 25 = minibatch size, 34 = max sentence length in words of the minibatch), the entries corresponding to scores of 0 are masked out (i.e. mapped to 0)

        #print('emb')
        #for i in range(34):
            #if emb[0][i].sum().item() != 0:
                #print(i)

        # z is also used to control when the encoder layer is active
        lengths = mask.long().sum(1)

        # encode the sentence
        _, final = self.enc_layer(emb, rnn_mask, lengths)

        # predict sentiment from final state(s)
        y = self.output_layer(final)

        return y
