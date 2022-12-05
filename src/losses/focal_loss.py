import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEFocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, alpha=0.25, reduction='mean'):
        super(BCEFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        probs = torch.sigmoid(inputs)  # sigmoide获取概率
        # 在原始ce上增加动态权重因子，注意alpha的写法，下面多类时不能这样使用
        loss = - self.alpha * (1 - probs) ** self.gamma * targets * torch.log(probs) - (
                    1 - self.alpha) * probs ** self.gamma * (
                       1 - targets) * torch.log(1 - probs)

        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss


class MultiFocalLoss(nn.Module):
    """
        Loss(x, class) = - \alpha*(1-softmax(x)[class])^gamma*log(softmax(x)[class])
        class_num：标签数目
        alpha：列表，各类的权重，比如[0.1, 0.2, 0.7]
        gamma：focal loss中的gamma参数
        size_average：损失计算，求和还是求平均
    """

    def __init__(self, class_num, alpha=None, gamma=2, reduction="mean"):
        super(MultiFocalLoss, self).__init__()
        self.class_num = class_num
        if alpha is None:
            alpha = torch.ones((class_num, 1), requires_grad=False)
        else:
            alpha = torch.tensor(alpha, requires_grad=False)

        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        batch_size = inputs.size(0)
        p = F.softmax(inputs, dim=-1)

        # 以下三行代码用于生成one-hot标签
        # 生成one-hot张量，但所有值都为0
        class_mask = inputs.data.new(batch_size, self.class_num).fill_(0)
        # 将标签转换为[batch_size, 1]
        ids = targets.view(-1, 1)

        # 这里注意scatter_的用法，
        # 1, ids.data, 1.：表示在dim=1上用ids.data作为索引填充1
        class_mask.scatter_(1, ids.data, 1.)

        alpha = self.alpha.to(inputs.device)
        # 根据真实标签索引选出某样本对应类别的权重
        alpha = alpha[ids.data.view(-1)]

        probs = (p * class_mask).sum(1).view(-1, 1)
        log_p = probs.log()

        batch_loss = - alpha * (torch.pow((1 - probs), self.gamma)) * log_p

        if self.reduction == "mean":
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()

        return loss


inputs = torch.rand((32, 3))
targets = torch.tensor([0, 1, 1, 0] * 8)
criterion = MultiFocalLoss(3)
loss = criterion(inputs, targets)
print(loss)

inputs = torch.rand((32))
targets = torch.tensor([0, 1] * 16)
criterion = BCEFocalLoss()
loss = criterion(inputs, targets)
print(loss)


class NerFocalLoss(nn.Module):
    """
        Loss(x, class) = - \alpha*(1-softmax(x)[class])^gamma*log(softmax(x)[class])
        class_num：标签数目
        alpha：列表，各类的权重，比如[0.1, 0.2, 0.7]
        gamma：focal loss中的gamma参数
        size_average：损失计算，求和还是求平均
    """

    def __init__(self, class_num, alpha=None, gamma=2, reduction="mean"):
        super(NerFocalLoss, self).__init__()
        self.class_num = class_num
        if alpha is None:
            alpha = torch.ones((class_num, 1), requires_grad=False)
        else:
            alpha = torch.tensor(alpha, requires_grad=False)

        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, attention_mask, targets):
        total = inputs.size(0) * inputs.size(1)
        active_loss = (attention_mask == 1).view(-1)
        inputs = inputs.view(-1, self.class_num)[active_loss]
        targets = targets.view(-1)[active_loss]

        p = F.softmax(inputs, dim=-1)
        # 以下三行代码用于生成one-hot标签
        # 生成one-hot张量，但所有值都为0
        class_mask = inputs.data.new(total, self.class_num).fill_(0)

        # 将标签转换为[batch_size, 1]
        ids = targets.view(total, 1).long()
        # 这里注意scatter_的用法，
        # 1, ids.data, 1.：表示在dim=1上用ids.data作为索引填充1
        class_mask.scatter_(1, ids.data, 1.)

        alpha = self.alpha.to(inputs.device)
        # 根据真实标签索引选出某样本对应类别的权重
        alpha = alpha[ids.data.view(-1)]

        probs = (p * class_mask).sum(1).view(-1, 1)
        log_p = probs.log()

        batch_loss = - alpha * (torch.pow((1 - probs), self.gamma)) * log_p

        if self.reduction == "mean":
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()

        return loss


inputs = torch.rand((32, 150, 3))
attention_mask = torch.ones((32, 150))
targets = torch.ones((32, 150))
criterion = NerFocalLoss(3)
loss = criterion(inputs, attention_mask, targets)
print(loss)
