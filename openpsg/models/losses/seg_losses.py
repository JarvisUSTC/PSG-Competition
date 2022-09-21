import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.builder import LOSSES
from mmdet.models.losses.utils import weighted_loss, weight_reduce_loss


#@mmcv.jit(derivate=True, coderize=True)
# @weighted_loss
# def dice_loss(input, target, mask=None, eps=0.001):
#     N, H, W = input.shape

#     input = input.contiguous().view(N, H * W)
#     target = target.contiguous().view(N, H * W).float()
#     if mask is not None:
#         mask = mask.contiguous().view(N, H * W).float()
#         input = input * mask
#         target = target * mask
#     a = torch.sum(input * target, 1)
#     b = torch.sum(input * input, 1) + eps
#     c = torch.sum(target * target, 1) + eps
#     d = (2 * a) / (b + c)
#     #print('1-d max',(1-d).max())
#     return 1 - d

def dice_loss(pred,
              target,
              weight=None,
              eps=1e-3,
              reduction='mean',
              naive_dice=False,
              avg_factor=None):
    """Calculate dice loss, there are two forms of dice loss is supported:

        - the one proposed in `V-Net: Fully Convolutional Neural
            Networks for Volumetric Medical Image Segmentation
            <https://arxiv.org/abs/1606.04797>`_.
        - the dice loss in which the power of the number in the
            denominator is the first power instead of the second
            power.

    Args:
        pred (torch.Tensor): The prediction, has a shape (n, *)
        target (torch.Tensor): The learning label of the prediction,
            shape (n, *), same shape of pred.
        weight (torch.Tensor, optional): The weight of loss for each
            prediction, has a shape (n,). Defaults to None.
        eps (float): Avoid dividing by zero. Default: 1e-3.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'.
            Options are "none", "mean" and "sum".
        naive_dice (bool, optional): If false, use the dice
                loss defined in the V-Net paper, otherwise, use the
                naive dice loss in which the power of the number in the
                denominator is the first power instead of the second
                power.Defaults to False.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    """

    input = pred.flatten(1)
    target = target.flatten(1).float()

    a = torch.sum(input * target, 1)
    if naive_dice:
        b = torch.sum(input, 1)
        c = torch.sum(target, 1)
        d = (2 * a + eps) / (b + c + eps)
    else:
        b = torch.sum(input * input, 1) + eps
        c = torch.sum(target * target, 1) + eps
        d = (2 * a) / (b + c)

    loss = 1 - d
    if weight is not None:
        assert weight.ndim == loss.ndim
        assert len(weight) == len(pred)
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


@LOSSES.register_module()
class psgtrDiceLoss(nn.Module):
    def __init__(self, eps=1e-6, reduction='mean', loss_weight=1.0):
        super(psgtrDiceLoss, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.count = 0

    def forward(self, inputs, targets, num_matches):
        inputs = inputs.sigmoid()
        inputs = inputs.flatten(1)
        targets = targets.flatten(1)
        numerator = 2 * (inputs * targets).sum(1)
        denominator = inputs.sum(-1) + targets.sum(-1)
        loss = 1 - (numerator + 1) / (denominator + 1)
        return self.loss_weight * loss.sum() / num_matches

@LOSSES.register_module()
class Hybrid_DiceLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=True,
                 activate=True,
                 reduction='mean',
                 naive_dice=False,
                 loss_weight=1.0,
                 eps=1e-3):
        """Compute dice loss.

        Args:
            use_sigmoid (bool, optional): Whether to the prediction is
                used for sigmoid or softmax. Defaults to True.
            activate (bool): Whether to activate the predictions inside,
                this will disable the inside sigmoid operation.
                Defaults to True.
            reduction (str, optional): The method used
                to reduce the loss. Options are "none",
                "mean" and "sum". Defaults to 'mean'.
            naive_dice (bool, optional): If false, use the dice
                loss defined in the V-Net paper, otherwise, use the
                naive dice loss in which the power of the number in the
                denominator is the first power instead of the second
                power. Defaults to False.
            loss_weight (float, optional): Weight of loss. Defaults to 1.0.
            eps (float): Avoid dividing by zero. Defaults to 1e-3.
        """

        super(Hybrid_DiceLoss, self).__init__()
        self.use_sigmoid = use_sigmoid
        self.reduction = reduction
        self.naive_dice = naive_dice
        self.loss_weight = loss_weight
        self.eps = eps
        self.activate = activate

    def forward(self,
                pred,
                target,
                weight=None,
                reduction_override=None,
                avg_factor=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction, has a shape (n, *).
            target (torch.Tensor): The label of the prediction,
                shape (n, *), same shape of pred.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction, has a shape (n,). Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum".

        Returns:
            torch.Tensor: The calculated loss
        """

        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        if self.activate:
            if self.use_sigmoid:
                pred = pred.sigmoid()
            else:
                raise NotImplementedError

        loss = self.loss_weight * dice_loss(
            pred,
            target,
            weight,
            eps=self.eps,
            reduction=reduction,
            naive_dice=self.naive_dice,
            avg_factor=avg_factor)

        return loss

@LOSSES.register_module()
class MultilabelCrossEntropy(torch.nn.Module):
    def __init__(self, reduction='mean', loss_weight=1.0):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, inputs, targets):
        assert (targets.sum(1) != 0).all()
        loss = -(F.log_softmax(inputs, dim=1) *
                 targets).sum(1) / targets.sum(1)
        loss = loss.mean()
        return self.loss_weight * loss


@LOSSES.register_module()
class MultilabelLogRegression(torch.nn.Module):
    def __init__(self, reduction='mean', loss_weight=1.0):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, inputs, targets):
        assert (targets.sum(1) != 0).all()
        loss_1 = -(torch.log((inputs + 1) / 2 + 1e-14) * targets).sum()
        loss_0 = -(torch.log(1 - (inputs + 1) / 2 + 1e-14) *
                   (1 - targets)).sum()
        # loss = loss.mean()
        return self.loss_weight * (loss_1 + loss_0) / (targets.sum() +
                                                       (1 - targets).sum())


@LOSSES.register_module()
class LogRegression(torch.nn.Module):
    def __init__(self, reduction='mean', loss_weight=1.0):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, inputs, targets):
        positive_rate = 50
        loss_1 = -(torch.log(
            (inputs + 1) / 2 + 1e-14) * targets).sum() * positive_rate
        loss_0 = -(torch.log(1 - (inputs + 1) / 2 + 1e-14) *
                   (1 - targets)).sum()
        return self.loss_weight * (loss_1 + loss_0) / (targets.sum() +
                                                       (1 - targets).sum())

    # def forward(self, inputs, targets):
    #     loss_1 = -(torch.log((inputs + 1) / 2 + 1e-14) * targets).sum()
    #     return self.loss_weight * loss_1

    # def forward(self, inputs, targets):
    #     inputs  = (inputs + 1) / 2 + 1e-14
    #     loss = F.mse_loss(inputs, targets.float(), reduction='mean')
    #     return self.loss_weight * loss


@LOSSES.register_module()
class BCEFocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, alpha=0.25, reduction='sum', loss_weight=1.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, inputs, targets, num_matches):

        prob = inputs.sigmoid()
        ce_loss = F.binary_cross_entropy_with_logits(inputs,
                                                     targets,
                                                     reduction='none')
        p_t = prob * targets + (1 - prob) * (1 - targets)
        loss = ce_loss * ((1 - p_t)**self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        return self.loss_weight * loss.mean(1).sum() / num_matches

        # pt = torch.sigmoid(_input)
        # bs = len(pt)
        # target = target.type(torch.long)
        # # print(pt.shape, target.shape)
        # alpha = self.alpha
        # loss = - alpha * (1 - pt) ** self.gamma * target * torch.log(pt) - \
        #     (1 - alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)
        # # print('loss_shape',loss.shape)
        # if self.reduction == 'elementwise_mean':
        #   loss = torch.mean(loss)
        # elif self.reduction == 'sum':
        #   loss = torch.sum(loss)

        # return loss*self.loss_weight/bs
