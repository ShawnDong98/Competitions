import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable


class FocalLossWithSmoothing(nn.Module):
    def __init__(
            self,
            num_classes: int,
            gamma: int = 2,
            lb_smooth: float = 0.1,
            ignore_index: int = None,
            alpha: float = None):
        """
        :param gamma:
        :param lb_smooth:
        :param ignore_index:
        :param size_average:
        :param alpha:
        """
        super(FocalLossWithSmoothing, self).__init__()
        if alpha is None:
            self._alpha = Variable(torch.ones(num_classes, 1))
        else:
            if isinstance(alpha, Variable):
                self._alpha = alpha
            else:
                self._alpha = Variable(alpha)

        self._num_classes = num_classes
        self._gamma = gamma
        self._lb_smooth = lb_smooth
        self._ignore_index = ignore_index
        self._log_softmax = nn.LogSoftmax(dim=1)

        if self._num_classes <= 1:
            raise ValueError('The number of classes must be 2 or higher')
        if self._gamma < 0:
            raise ValueError('Gamma must be 0 or higher')
        if alpha is not None:
            if alpha <= 0 or alpha >= 1:
                raise ValueError('Alpha must be 0 <= alpha <= 1')

    def forward(self, logits, label):
        """
        :param logits: (batch_size, class, height, width)
        :param label:
        :return:
        """
        logits = logits.float()
        difficulty_level = self._estimate_difficulty_level(logits, label)

        with torch.no_grad():
            label = label.clone().detach()
            if self._ignore_index is not None:
                ignore = label.eq(self._ignore_index)
                label[ignore] = 0
            lb_pos, lb_neg = 1. - self._lb_smooth, self._lb_smooth / (self._num_classes - 1)
            lb_one_hot = torch.empty_like(logits).fill_(
                lb_neg).scatter_(1, label.unsqueeze(1), lb_pos).detach()
        if logits.is_cuda and not self._alpha.is_cuda:
            self._alpha = self._alpha.cuda()
        alpha = self._alpha[label.data.view(-1)]
        
        logs = self._log_softmax(logits)
        loss = -alpha * torch.sum(difficulty_level * logs * lb_one_hot, dim=1)
        if self._ignore_index is not None:
            loss[ignore] = 0
        return loss.mean()

    def _estimate_difficulty_level(self, logits, label):
        """
        :param logits:
        :param label:
        :return:
        """
        one_hot_key = torch.nn.functional.one_hot(label, num_classes=self._num_classes)
        if len(one_hot_key.shape) == 4:
            one_hot_key = one_hot_key.permute(0, 3, 1, 2)
        if one_hot_key.device != logits.device:
            one_hot_key = one_hot_key.to(logits.device)
        pt = one_hot_key * F.softmax(logits, dim=1)
        difficulty_level = torch.pow(1 - pt, self._gamma)
        return difficulty_level

class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in 
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5), 
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """
    def __init__(self, class_num=200, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs,dim=1)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        #print(class_mask)


        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P*class_mask).sum(1).view(-1,1)

        log_p = probs.log()
        #print('probs size= {}'.format(probs.size()))
        #print(probs)

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p 

        #print('-----bacth_loss------')
        #print(batch_loss)


        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss

class focal_loss(nn.Module):
    def __init__(self):
        super(focal_loss, self).__init__()
        self.gamma = 2.
        self.alpha=.25 

    def forward(self, y_true, y_pred):
        # Scale predictions so that the class probas of each sample sum to 1
        y_pred /= torch.sum(y_pred, axis=-1, keepdims=True)  
        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = 1e-7
        y_pred = torch.clip(y_pred, epsilon, 1. - epsilon)
        # Calculate Cross Entropy
        cross_entropy = -y_true * torch.log(y_pred)
        # Calculate Focal Loss
        loss = self.alpha * torch.pow(1 - y_pred, self.gamma) * cross_entropy
        # Compute mean loss in mini_batch
        return torch.mean(loss, axis=1)


class focal_f1_loss(nn.Module):  
    def __init__(self):
        super(focal_f1_loss, self).__init__()
        self.alpha =  0.001
        self.focal_loss = FocalLoss()
    
    def forward(self, y_pred, y_true):
        N = y_pred.size(0)
        C = y_pred.size(1)
        class_mask = y_pred.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = y_true.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)

        tp = torch.sum(class_mask*y_pred, axis=0)
        tn = torch.sum((1-class_mask)*(1-y_pred), axis=0)
        fp = torch.sum((1-class_mask)*y_pred, axis=0)
        fn = torch.sum(class_mask*(1-y_pred), axis=0)
        p = tp / (tp + fp + 1e-7)
        r = tp / (tp + fn + 1e-7)
        f1 = 2*p*r / (p+r+1e-7)

        return self.alpha *  (1 - torch.mean(f1)) +  self.focal_loss(y_pred, y_true)

  