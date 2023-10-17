import numpy as np
import torch
import torch.nn as nn
from medpy.metric.binary import dc

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target):
        N = target.size(0)
        smooth = 1e-5

        input_flat = input.view(N, -1)
        target_flat = target.view(N, -1)

        intersection = input_flat * target_flat

        dice = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        loss = 1 - dice.mean()

        return loss

def one_hot_encoder(x, num_classes, on_value=1., off_value=0., device='cuda:1'):
    #x = x.unsqueeze(1)
    return torch.full((x.size()[0], num_classes,x.size()[2],x.size()[2]), off_value, device=device).scatter_(1, x, on_value)


class MultiClassDiceLoss(nn.Module):
    """
    requires one hot encoded target. Applies DiceLoss on each class iteratively.
    requires input.shape[0:1] and target.shape[0:1] to be (N, C) where N is
      batch size and C is number of classes
    """

    def __init__(self):
        super(MultiClassDiceLoss, self).__init__()

    def forward(self, input, target,weights=None):

        C = input.shape[1]

        target_oh = one_hot_encoder(target.long(), C)
        dice = DiceLoss()
        totalLoss = 0
        allLoss = []
        totalDice = 0

        for i in range(C):
            diceLoss = dice(input[:, i], target_oh[:, i])
            dice_core = 1-diceLoss
            totalDice = totalDice+dice_core
            #metric_loss =1-dc(pred_oh[:,i].cpu().numpy(),target_oh[:,i].cpu().numpy())
            if weights is not None:
                diceLoss *= weights[i]
            totalLoss = totalLoss + diceLoss
            allLoss.append(diceLoss.item())

        return totalLoss / C, totalDice / C


class Dice(nn.Module):
    def __init__(self):
        super(Dice, self).__init__()

    def forward(self, input, target):

        #N = target.size(0)
        N = target.shape[0]
        smooth = 1e-5

        input_flat = input.view(N, -1)
        target_flat = target.view(N, -1)

        intersection = input_flat * target_flat

        dice = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        dice_mean = dice.mean()

        return dice_mean


class MultiClassDice(nn.Module):
    """
    requires one hot encoded target. Applies DiceLoss on each class iteratively.
    requires input.shape[0:1] and target.shape[0:1] to be (N, C) where N is
      batch size and C is number of classes
    """

    def __init__(self):
        super(MultiClassDice, self).__init__()

    def forward(self, input, target, weights=None):
        #可取当前mask所代表的值
        C = input.shape[1]

        # if weights is None:
        # 	weights = torch.ones(C) #uniform weights for all classes

        #可取当前mask所代表的值
        #target[target > 0].data[0]

        dice = Dice()
        totalDice = []

        target = one_hot_encoder(target.long(), C)

        for i in range(C):
            metricDice = dice(input[:, i], target[:, i])
            if weights is not None:
                metricDice *= weights[i]
            totalDice.append(metricDice.item())

        return sum(totalDice)/C,totalDice

class GlobalDiceLoss(nn.Module):
    """
    requires one hot encoded target. Applies DiceLoss on each class iteratively.
    requires input.shape[0:1] and target.shape[0:1] to be (N, C) where N is
      batch size and C is number of classes
    """

    def __init__(self):
        super(GlobalDiceLoss, self).__init__()
        self.weights = [0.5, 0.5]

    def forward(self, input, target, weights=None):
        #可取当前mask所代表的值
        target_num = target[target > 0].data[0]
        preds = torch.max(input,dim=1).indices
        preds[preds < target_num] = 0
        preds[preds > target_num] = 0
        preds[preds == target_num] = 1
        logit = preds.unsqueeze(1)
        smooth = 1e-5
        batch_size = len(logit)
        logit = logit.view(batch_size,-1)
        truth = target.view(batch_size,-1)
        assert(logit.shape==truth.shape)
        p = logit.view(batch_size,-1)
        t = truth.view(batch_size,-1)
        w = truth.detach()
        w = w*(self.weights[1]-self.weights[0])+self.weights[0]
        p = w*(p)
        t = w*(t)
        intersection = (p * t).sum(-1)
        union =  (p * p).sum(-1) + (t * t).sum(-1)
        dice  = 1 - (2*intersection + smooth) / (union +smooth)

        loss = dice.mean()
        return loss

class Marginal_Loss(nn.Module):
    def __init__(self):
        super(Marginal_Loss, self).__init__()

    def forward(self, pred, target, client_id):
        if client_id == 0:
            #0
            class_flag = [0,1,2]
            #class_flag = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
            # C = input.shape[1]
        # if client_id == 1:
        #     #0 1
        #     class_flag = [1, 4, 5, 6,7,10]
        #     #class_flag = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        #     # C = input.shape[1]
        # if client_id == 2:
        #     #class_flag = [0, 8, 9]
        #     class_flag = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        # if client_id == 999:
        else:
            class_flag = [0,1,2,3,4]
        #边际处理过后的预测的标签
        margin_prob = merge_prob(pred, class_flag)
        _,C,_,_ = margin_prob.size()
        # dice_loss = dice_loss_fn(
        #     margin_prob.float(),
        #     F.one_hot(target.squeeze(1).to(torch.int64), C).permute(0, 3, 1, 2).float(),
        #     multiclass=True
        # )
        dice_loss,dice_score = MultiClassDiceLoss()(margin_prob, target,client_id)
        margin_log_prob = torch.log(torch.clamp(margin_prob, min=1e-4))
        ce_loss = nn.NLLLoss()(margin_log_prob,target[:,0,:,:].long())

        #这里修改了
        # margin_target = merge_label(target, class_flag)
        # margin_log_prob = torch.log(torch.clamp(margin_prob, min=1e-4))
        # ce_loss = nn.NLLLoss()
        # l_ce = ce_loss(margin_log_prob, margin_target.long().squeeze(dim=1))
        # margin_target_oh = one_hot_encoder(margin_target.long(), len(class_flag))
        # mul_dice_loss = WeightDiceLoss()
        # l_dice = mul_dice_loss(margin_prob, margin_target_oh)
        return ce_loss, dice_loss
def merge_prob(prob, class_flag):
    bg_prob_list = []
    # for c, class_exist in enumerate(class_flag):
    #     if c == 0 or class_exist == 0:
    #         bg_prob_list.append(prob[:,c:c+1,:])
    #bg_prob_list 背景的list
    for c in range(5):
        if c == 0:
            bg_prob_list.append(prob[:,0:1,:])
        else:
            if c not in class_flag:
                bg_prob_list.append(prob[:, c:c+1, :])
    #合并背景
    bg_prob = torch.sum(torch.cat(bg_prob_list, dim=1), dim=1, keepdim=True)
    #bg_prob是新的背景
    merged_prob_list = [bg_prob]
    #寻找存在的标签
    for c, class_exist in enumerate(class_flag):
        if c > 0 and class_exist > 0:
            merged_prob_list.append(prob[:,class_exist:class_exist+1,:])
    margin_prob = torch.cat(merged_prob_list, dim=1)
    return margin_prob



class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1, p=2):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p

    def forward(self, predict, target, flag):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        intersection = self.smooth
        union = self.smooth
        if flag is None:
            pd = predict
            gt = target
            intersection += torch.sum(pd*gt)*2
            union += torch.sum(pd.pow(self.p) + gt.pow(self.p))
        else:
            for i in range(target.shape[0]):
                if flag[i,0] > 0:
                    pd = predict[i:i+1,:]
                    gt = target[i:i+1,:]
                    intersection += torch.sum(pd*gt)*2
                    union += torch.sum(pd.pow(self.p) + gt.pow(self.p))
        dice = intersection / union

        loss = 1 - dice
        return loss

"""
# 多分类的 FocalLoss
如果是二分类问题，alpha 可以设置为一个值
如果是多分类问题，这里只能取list 外面要设置好list 并且长度要与分类bin大小一致,并且alpha的和要为1  
比如dist 的alpha=[0.02777]*36 +[0.00028] 这里是37个分类，设置前36个分类系数一样，最后一个分类权重系数小于前面的。

注意: 这里默认 input是没有经过softmax的，并且把shape 由H*W 2D转成1D的，然后再计算
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Focal_Loss(nn.Module):
    def __init__(self, weight=1, gamma=2):
        super(Focal_Loss, self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, preds, labels):
        """
        preds:softmax输出结果
        labels:真实值
        """
        C = preds.shape[1]
        labels = one_hot_encoder(labels.long(), C)
        eps = 1e-7
        y_pred = preds.view((preds.size()[0], preds.size()[1], -1))  # B*C*H*W->B*C*(H*W)

        target = labels.view(y_pred.size())  # B*C*H*W->B*C*(H*W)

        ce = -1 * torch.log(y_pred + eps) * target
        floss = torch.pow((1 - y_pred), self.gamma) * ce
        floss = torch.mul(floss, self.weight)
        floss = torch.sum(floss, dim=1)
        loss = torch.mean(floss)
        return loss

class FocalLoss(nn.Module):
    """
    copy from: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/FocalLoss/FocalLoss.py
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(self, apply_nonlin=None, alpha=[[0.1],[0.2],[0,1],[0.3],[0.3]], gamma=2, balance_index=0, smooth=1e-5, size_average=True):
        super(Focal_Loss, self).__init__()
        self.apply_nonlin = apply_nonlin
        self.alpha = None
        self.gamma = gamma
        self.balance_index = balance_index
        self.smooth = smooth
        self.size_average = size_average

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, logit, target):
        if self.apply_nonlin is not None:
            logit = self.apply_nonlin(logit)
        num_class = logit.shape[1]

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = torch.squeeze(target, 1)
        target = target.view(-1, 1)
        #
        alpha = self.alpha

        if alpha is None:
            alpha = torch.ones(num_class, 1)
        elif isinstance(alpha, (list, np.ndarray)):
            assert len(alpha) == num_class
            alpha = torch.FloatTensor(alpha).view(num_class, 1)
            alpha = alpha / alpha.sum()
        elif isinstance(alpha, float):
            alpha = torch.ones(num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[self.balance_index] = self.alpha

        else:
            raise TypeError('Not support alpha type')

        if alpha.device != logit.device:
            alpha = alpha.to(logit.device)

        idx = target.cpu().long()

        one_hot_key = torch.FloatTensor(target.size(0), num_class).zero_()
        #one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth / (num_class - 1), 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + self.smooth
        logpt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]
        alpha = torch.squeeze(alpha)
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss

class FocalLoss_Ori(nn.Module):
    """
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
    Focal_Loss= -1*alpha*((1-pt)**gamma)*log(pt)
    Args:
        num_class: number of classes
        alpha: class balance factor
        gamma:
        ignore_index:
        reduction:
    """

    def __init__(self, num_class=5, alpha=None, gamma=2, ignore_index=None, reduction='mean'):
        super(FocalLoss_Ori, self).__init__()
        self.num_class = num_class
        self.gamma = gamma
        self.reduction = reduction
        self.smooth = 1e-4
        self.ignore_index = ignore_index
        self.alpha = alpha
        if alpha is None:
            self.alpha = torch.ones(num_class, )
        elif isinstance(alpha, (int, float)):
            self.alpha = torch.as_tensor([alpha] * num_class)
        elif isinstance(alpha, (list, np.ndarray)):
            self.alpha = torch.as_tensor(alpha)
        if self.alpha.shape[0] != num_class:
            raise RuntimeError('the length not equal to number of class')

    def forward(self, logit, target):
        # assert isinstance(self.alpha,torch.Tensor)\
        N, C = logit.shape[:2]
        alpha = self.alpha.to(logit.device)
        prob = F.softmax(logit, dim=1)
        if prob.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            prob = prob.view(N, C, -1)
            prob = prob.transpose(1, 2).contiguous()  # [N,C,d1*d2..] -> [N,d1*d2..,C]
            prob = prob.view(-1, prob.size(-1))  # [N,d1*d2..,C]-> [N*d1*d2..,C]
        ori_shp = target.shape
        target = target.view(-1, 1)  # [N,d1,d2,...]->[N*d1*d2*...,1]
        valid_mask = None
        if self.ignore_index is not None:
            valid_mask = target != self.ignore_index
            target = target * valid_mask
        target = target.long()
        # ----------memory saving way--------
        prob = prob.gather(1, target).view(-1) + self.smooth  # avoid nan
        logpt = torch.log(prob)
        # alpha_class = alpha.gather(0, target.view(-1))
        alpha_class = alpha[target.squeeze().long()]
        class_weight = -alpha_class * torch.pow(torch.sub(1.0, prob), self.gamma)
        loss = class_weight * logpt
        if valid_mask is not None:
            loss = loss * valid_mask.squeeze()

        if self.reduction == 'mean':
            loss = loss.mean()
            if valid_mask is not None:
                loss = loss.sum() / valid_mask.sum()
        elif self.reduction == 'none':
            loss = loss.view(ori_shp)
        return loss

