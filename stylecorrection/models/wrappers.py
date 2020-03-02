import torch.nn as nn


class DataParallelCELWrapper(nn.Module):

    def __init__(self,
                 model: nn.Module,
                 loss: nn.CrossEntropyLoss,
                 num_classes: int):
        super(DataParallelCELWrapper, self).__init__()
        self.model = model
        self.loss = loss
        self.num_classes = num_classes

    def forward(self, targets, *inputs):
        model_output = self.model(*inputs)
        return self.loss(model_output.contiguous().view(-1, self.num_classes), targets).unsqueeze(0)
