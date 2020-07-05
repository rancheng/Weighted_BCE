class WeightedBCELoss(BCELoss):
    def __init__(self):
        super(WeightedBCELoss, self).__init__()
        self.weight = torch.tensor([1 / 6, 5 / 6]).float().cuda()
        # weight class_0 as 1/6 and class_1 as 5/6

    def forward(self, output, label):
        batch_weight = self.weight[label.data.view(-1).long()].view_as(label)
        bce = torch.nn.BCELoss(weight=batch_weight)
        loss = bce(output, label)
        return loss
