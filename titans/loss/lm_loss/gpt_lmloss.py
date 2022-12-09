from torch import nn

from colossalai import nn as col_nn


class GPTLMLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.loss = col_nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        print('---------------------shift_logits.view(-1, shift_logits.size(-1)) : {}'.format((shift_logits.view(-1, shift_logits.size(-1))).shape))
        print('---------------------shift_labels.view(-1) : {}'.format((shift_labels.view(-1)).shape))
        return self.loss(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))