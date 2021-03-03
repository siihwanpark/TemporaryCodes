from __future__ import print_function

import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from distiller_zoo.OURS import D, prediction_MLP
from torch.optim.lr_scheduler import _LRScheduler
from itertools import chain

class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]

class TwoCropTransformWithBase:
    """Create two crops of the same image with original image"""
    def __init__(self, base_transform, transform):
        self.base_transform = base_transform
        self.transform = transform

    def __call__(self, x):
        return [self.base_transform(x), self.transform(x), self.transform(x)]

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

################################################################################################################################################

class SimCLRLoss(nn.Module):
    """ SimCLR Loss """
    def __init__(self, temperature=0.07):
        super(SimCLRLoss, self).__init__()
        self.temperature = temperature
        
    def forward(self, features):
        # batch = [x1, ... ,xN] --aug--> [x1, x2, ... ,x2N-1 x2N]
        # x1 -> x1, xN+1 / xk -> xk, xN+k
        # features = [f(x1), f(x2), ... , f(x2N)]; [2N, s_dim]
        
        N = features.size(0) // 2

        anchor = F.normalize(features, dim=-1)
        contrast = anchor

        product = torch.mm(anchor, contrast.T) # [2N, 2N]
        sim_matrix = torch.exp(product / self.temperature)

        block1 = torch.eye(N)
        block2 = torch.eye(2*N)

        pos_mask = (block1.repeat(2, 2) - block2).cuda() # [2N, 2N]; masking positive pairs
        neg_mask = (torch.ones(2*N, 2*N) - block2).cuda() # [2N, 2N]; mask out self-contrast cases

        numerator = torch.sum(sim_matrix * pos_mask, dim=-1) # [2N, 1]
        denominator = torch.sum(sim_matrix * neg_mask, dim=-1) # [2N, 1]

        l = -torch.log(numerator / denominator) # [2N, 1]
        loss = l.mean()

        return loss

################################################################################################################################################s

class SimSiamLoss(nn.Module):
    """ SimSiam Loss """
    def __init__(self, opt):
        super(SimSiamLoss, self).__init__()
        self.predictor = prediction_MLP(in_dim=opt.s_dim, hidden_dim=opt.h_dim,
                                        out_dim=opt.s_dim)
        
    def forward(self, features):
        bsz = features.size(0) // 2
        z1, z2 = torch.split(features, [bsz, bsz], dim=0)
        p1, p2 = self.predictor(z1), self.predictor(z2)
        return D(p1, z2) / 2 + D(p2, z1) / 2

def norm_mse_loss(x0, x1):
    x0 = F.normalize(x0)
    x1 = F.normalize(x1)
    return 2 - 2 * (x0 * x1).sum(dim=-1).mean()

################################################################################################################################################

class BYOL(nn.Module):
    """ implements BYOL loss https://arxiv.org/abs/2006.07733 """

    def __init__(self, opt, model):
        """ init additional target and predictor networks """
        super(BYOL, self).__init__()
        self.model = model()

        self.head = nn.Sequential(
            nn.Linear(opt.s_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 64)
        )

        self.pred = nn.Sequential(
            nn.Linear(64, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 64),
        )
        self.model_t = model()
        self.head_t = nn.Sequential(
            nn.Linear(opt.s_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 64)
        )

        for param in chain(self.model_t.parameters(), self.head_t.parameters()):
            param.requires_grad = False
        self.update_target(0)
        self.byol_tau = 0.99
        self.loss_f = norm_mse_loss

    def update_target(self, tau):
        """ copy parameters from main network to target """
        for t, s in zip(self.model_t.parameters(), self.model.parameters()):
            t.data.copy_(t.data * tau + s.data * (1.0 - tau))
        for t, s in zip(self.head_t.parameters(), self.head.parameters()):
            t.data.copy_(t.data * tau + s.data * (1.0 - tau))

    def forward(self, samples):
        z = [self.pred(self.head(self.model(x, is_feat=True)[-1])) for x in samples]
        with torch.no_grad():
            zt = [self.head_t(self.model_t(x, is_feat=True)[-1]) for x in samples]

        loss = 0
        for i in range(len(samples) - 1):
            for j in range(i + 1, len(samples)):
                loss += self.loss_f(z[i], zt[j]) + self.loss_f(z[j], zt[i])
        loss /= 2
        return loss

    def step(self, progress):
        """ update target network with cosine increasing schedule """
        tau = 1 - (1 - self.byol_tau) * (math.cos(math.pi * progress) + 1) / 2
        self.update_target(tau)

class BYOL_distill(nn.Module):
    """ implements BYOL loss https://arxiv.org/abs/2006.07733 """

    def __init__(self, opt, model_func, teacher):
        """ init additional target and predictor networks """
        super(BYOL_distill, self).__init__()
        self.teacher = teacher
        self.model = model_func()

        self.head = nn.Sequential(
            nn.Linear(opt.s_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 64)
        )

        self.pred = nn.Sequential(
            nn.Linear(64, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 64),
        )
        self.model_t = model_func()
        self.head_t = nn.Sequential(
            nn.Linear(opt.s_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 64)
        )

        # For Distillation
        self.predictor = prediction_MLP(in_dim=opt.s_dim,
                            hidden_dim=opt.h_dim, out_dim=opt.t_dim)

        for param in chain(self.model_t.parameters(), self.head_t.parameters(), self.teacher.parameters()):
            param.requires_grad = False
        
        self.update_target(0)
        self.byol_tau = 0.99
        self.loss_f = norm_mse_loss

    def update_target(self, tau):
        """ copy parameters from main network to target """
        for t, s in zip(self.model_t.parameters(), self.model.parameters()):
            t.data.copy_(t.data * tau + s.data * (1.0 - tau))
        for t, s in zip(self.head_t.parameters(), self.head.parameters()):
            t.data.copy_(t.data * tau + s.data * (1.0 - tau))

    def forward(self, samples):
        z = [self.pred(self.head(self.model(x, is_feat=True)[-1])) for x in samples]
        with torch.no_grad():
            zt = [self.head_t(self.model_t(x, is_feat=True)[-1]) for x in samples]

        loss = 0
        for i in range(len(samples) - 1):
            for j in range(i + 1, len(samples)):
                loss += self.loss_f(z[i], zt[j]) + self.loss_f(z[j], zt[i])
        loss /= 2

        z1, z2 = self.model(samples[0], is_feat=True)[-1], self.model(samples[1], is_feat=True)[-1]
        p1, p2 = self.predictor(z1), self.predictor(z2)
        t1, t2 = self.teacher(samples[0], is_feat=True)[-1], self.teacher(samples[1], is_feat=True)[-1]

        loss_kd = D(p1, t1) / 2 + D(p2, t2) / 2

        return loss, loss_kd

    def step(self, progress):
        """ update target network with cosine increasing schedule """
        tau = 1 - (1 - self.byol_tau) * (math.cos(math.pi * progress) + 1) / 2
        self.update_target(tau)

################################################################################################################################################
# MoCo

class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, dim=512, K=65536, m=0.999, T=0.07, mlp=False):
        """
        dim: feature dimension (default: 512)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder()
        self.encoder_k = base_encoder()

        self.head_q = nn.Sequential(nn.Linear(dim, dim), nn.ReLU())
        self.head_k = nn.Sequential(nn.Linear(dim, dim), nn.ReLU())

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        for param_q, param_k in zip(self.head_q.parameters(), self.head_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

        for param_q, param_k in zip(self.head_q.parameters(), self.head_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        #assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        if ptr + batch_size > self.K:
            rounded = ptr + batch_size - self.K
            self.queue[:, ptr:] = keys[:batch_size-rounded, :].T
            self.queue[:, :rounded] = keys[batch_size-rounded:, :].T
            ptr = rounded
        else:
            self.queue[:, ptr:ptr + batch_size] = keys.T
            ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    def forward(self, im_q, im_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        q = self.head_q(self.encoder_q(im_q, is_feat=True)[-1])  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            k = self.head_k(self.encoder_k(im_k, is_feat=True)[-1])  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels

class MoCo_distill(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, opt, base_encoder, teacher, K=65536, m=0.999, T=0.07, mlp=False):
        """
        dim: feature dimension (default: 512)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo_distill, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # distill setting
        self.teacher = teacher
        self.predictor = prediction_MLP(in_dim=opt.s_dim,
                            hidden_dim=opt.h_dim, out_dim=opt.t_dim)

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder()
        self.encoder_k = base_encoder()

        self.head_q = nn.Sequential(nn.Linear(opt.s_dim, opt.s_dim), nn.ReLU())
        self.head_k = nn.Sequential(nn.Linear(opt.s_dim, opt.s_dim), nn.ReLU())

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        for param_q, param_k in zip(self.head_q.parameters(), self.head_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        for param in self.teacher.parameters():
            param.requires_grad = False # not updated by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(opt.s_dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

        for param_q, param_k in zip(self.head_q.parameters(), self.head_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        #assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        if ptr + batch_size > self.K:
            rounded = ptr + batch_size - self.K
            self.queue[:, ptr:] = keys[:batch_size-rounded, :].T
            self.queue[:, :rounded] = keys[batch_size-rounded:, :].T
            ptr = rounded
        else:
            self.queue[:, ptr:ptr + batch_size] = keys.T
            ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    def forward(self, im_q, im_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        q = self.head_q(self.encoder_q(im_q, is_feat=True)[-1])  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            k = self.head_k(self.encoder_k(im_k, is_feat=True)[-1])  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        # distllation loss
        z1, z2 = self.encoder_q(im_q, is_feat=True)[-1], self.encoder_q(im_k, is_feat=True)[-1]
        p1, p2 = self.predictor(z1), self.predictor(z2)
        t1, t2 = self.teacher(im_q, is_feat=True)[-1], self.teacher(im_k, is_feat=True)[-1]

        loss_kd = D(p1, t1) / 2 + D(p2, t2) / 2

        return [logits, labels], loss_kd
################################################################################################################################################





def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def save_model(model, optimizer, opt, epoch, save_file):
    print('==> Saving...')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    print('Checkpoint saved to ' + save_file)
    del state
