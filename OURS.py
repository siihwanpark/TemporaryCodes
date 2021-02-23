from __future__ import print_function

import torch.nn as nn
import torch.nn.functional as F

def D(p, z, weight=None, version='simplified'): # negative cosine similarity
    # weight : weights for weighted sum with size [batch]

    if version == 'original':
        z = z.detach() # stop gradient
        p = F.normalize(p, dim=1) # l2-normalize 
        z = F.normalize(z, dim=1) # l2-normalize 
        return -(p*z).sum(dim=1).mean()

    elif version == 'simplified':# same thing, much faster. Scroll down, speed test in __main__
        if weight is None:
            return - F.cosine_similarity(p, z.detach(), dim=-1).mean()
        else:
            loss = - F.cosine_similarity(p, z.detach(), dim=-1) # [batch]
            return (weight * loss).mean()

class prediction_MLP(nn.Module):
    def __init__(self, in_dim=2048, hidden_dim=512, out_dim=2048): # bottleneck structure
        super().__init__()
        ''' page 3 baseline setting
        Prediction MLP. The prediction MLP (h) has BN applied 
        to its hidden fc layers. Its output fc does not have BN
        (ablation in Sec. 4.4) or ReLU. This MLP has 2 layers. 
        The dimension of h’s input and output (z and p) is d = 2048, 
        and h’s hidden layer’s dimension is 512, making h a 
        bottleneck structure (ablation in supplement). 
        '''
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)
        """
        Adding BN to the output of the prediction MLP h does not work
        well (Table 3d). We find that this is not about collapsing. 
        The training is unstable and the loss oscillates.
        """

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x 

class OURS(nn.Module):
    """ Ours : SimSiam-based distillation """
    def __init__(self, opt):
        super(OURS, self).__init__()
        self.predictor = prediction_MLP(in_dim=opt.s_dim,
                            hidden_dim=opt.h_dim, out_dim=opt.t_dim)
        
        self.weighted = opt.weighted
        self.weighting = opt.weighting

    def forward(self, s1, s2, t1, t2):
        p1, p2 = self.predictor(s1), self.predictor(s2)
        
        weights = None

        if self.weighted:
            #t_weights = torch.exp(F.cosine_similarity(t1, t2, dim=-1)) # [batch]

            f1, f2 = self.backbone(x1), self.backbone(x2)
            s_weights = torch.exp(F.cosine_similarity(f1, f2, dim=-1))

            weights = 1 / s_weights
            
            if self.weighting == 'naive':
                pass
            elif self.weighting == 'min-max':
                maximum = weights.max()
                minimum = weights.min() 
                minmax = maximum - minimum
                weights = weights.div(minmax.expand_as(weights))
            elif self.weighting == 'l1_normalized':
                l1_norm = torch.norm(weights, p=1).detach()
                weights = weights.div(l1_norm.expand_as(weights))
            elif self.weighting == 'l2_normalized':
                l2_norm = torch.norm(weights, p=2).detach()
                weights = weights.div(l2_norm.expand_as(weights))
            elif self.weighting == 'softmax':
                weights = F.softmax(weights, dim=-1)
            else:
                raise ValueError(self.weighting)

        loss = (D(p1, t1, weight=weights) + D(p2, t2, weight=weights))/2
        return loss

class SimCLRLoss(nn.Module):
    """ SimCLR Loss """
    def __init__(self, opt):
        super(SimlrDistillv1, self).__init__()
        self.temperature = 1.0

    def forward(self, features):
        # batch = [x1, ... ,xN] --aug--> [x1, x2, ... ,x2N-1 x2N]
        # x1 -> x1, xN+1 / xk -> xk, xN+k
        # features = [f_s(x1), f_s(x2), ... , f_s(x2N)]; [2N, s_dim]
        
        N = features.size(0) // 2

        anchor = features
        contrast = features

        contrast_T = contrast.transpose(0, 1) # [s_dim, 2N]
        
        product = torch.mm(anchor, contrast_T) # [2N, 2N]

        anchor_norm = torch.norm(anchor, dim=-1).unsqueeze(-1) # [2N, 1]
        contrast_norm = torch.norm(contrast, dim=-1).unsqueeze(-1) # [2N, 1]
        norm_matrix = torch.mm(anchor_norm, contrast_norm.transpose(0, 1)) # [2N, 1] * [1, 2N] -> [2N, 2N]
        
        sim_matrix = product / norm_matrix
        sim_matrix = sim_matrix / self.temperature
        sim_matrix = torch.exp(sim_matrix)

        block1 = torch.zeros(N, N)
        block2 = torch.eyes(N)

        block3 = torch.cat([block1, block2], dim=0) # [2N, N]
        block4 = torch.cat([block2, block1], dim=0) # [2N, N]

        pos_mask = torch.cat([block3, block4], dim=-1) # [2N, 2N]
        neg_mask = torch.ones(2*N, 2*N) - torch.eyes(2*N) # [2N, 2N]

        numerator = sim_matrix * pos_mask
        denominator = sim_matrix * neg_mask

        numerator = torch.sum(numerator, dim=-1) # [2N, 1]
        denominator = torch.sum(denominator, dim=-1) # [2N, 1]

        l = -torch.log(numerator / denominator) # [2N, 1]
        loss = l.mean()

        return loss

class SimclrDistillv1(nn.Module):
    """ OURS : SimCLR-base distillation v1"""
    def __init__(self, opt):
        super(SimlrDistillv1, self).__init__()
        self.predictor = prediction_MLP(in_dim=opt.s_dim,
                            hidden_dim=opt.h_dim, out_dim=opt.t_dim)
        
        self.temperature = 1.0

    def forward(self, f_s, f_t):
        # batch = [x1, ... ,xN] --aug--> [x1, x2, ... ,x2N-1 x2N]
        # x1 -> x1, xN+1 / xk -> xk, xN+k
        # f_s = [f_s(x1), f_s(x2), ... , f_s(x2N)]; [2*batch, s_dim]
        # f_t = [f_t(x1), f_t(x2), ... , f_t(x2N)]; [2*batch, t_dim]
        
        N = f_s.size(0) // 2

        f_s = self.predictor(f_s) # [2*batch, s_dim] -> [2*batch, t_dim]
        features = torch.cat([f_s, f_t], dim=0) # [4*batch, t_dim]
        features_T = features.transpose(0, 1) # [t_dim, 4*batch]
        
        product = torch.mm(features, features_T) # [4N, 4N]

        norm = torch.norm(features, dim=-1).unsqueeze(-1) # [4N, 1]
        norm_matrix = torch.mm(norm, norm.transpose(0, 1)) # [4N, 1] * [1, 4N] -> [4N, 4N]
        
        sim_matrix = product / norm_matrix
        sim_matrix = sim_matrix / self.temperature
        sim_matrix = torch.exp(sim_matrix)

        block1 = torch.zeros(N, N) # [N, N]
        block2 = torch.eyes(N) # [N, N]
        
        block3 = torch.cat([block1, block2], dim=0) # [2N, N]
        block4 = torch.cat([block2, block1], dim=0) # [2N, N]
        
        block5 = torch.cat([block3, block4], dim=1) # [2N, 2N]
        block6 = block2.repeat(2, 2) # [2N, 2N]
        
        block7 = torch.cat([block5, block6], dim=0) # [4N, 2N]
        block8 = torch.cat([block6, block5], dim=0) # [4N, 2N]

        pos_mask = torch.cat([block7, block8], dim=1) # [4N, 4N]
        neg_mask = torch.ones(4*N, 4*N) - torch.eyes(4*N) # [4N, 4N]

        numerator = sim_matrix * pos_mask
        denominator = sim_matrix * neg_mask

        numerator = torch.sum(numerator, dim=-1) # [4N, 1]
        denominator = torch.sum(denominator, dim=-1) # [4N, 1]

        l = -torch.log(numerator / denominator) # [4N, 1]

        loss = l.mean()
        return loss

class SimclrDistillv2(nn.Module):
    """ OURS : SimCLR-base distillation v2"""
    def __init__(self, opt):
        super(SimlrDistillv1, self).__init__()
        self.predictor = prediction_MLP(in_dim=opt.s_dim,
                            hidden_dim=opt.h_dim, out_dim=opt.t_dim)
        
        self.temperature = 1.0

    def forward(self, f_s, f_t):
        # batch = [x1, ... ,xN] --aug--> [x1, x2, ... ,x2N-1 x2N]
        # x1 -> x1, xN+1 / xk -> xk, xN+k
        # f_s = [f_s(x1), f_s(x2), ... , f_s(x2N)]; [2*batch, s_dim]
        # f_t = [f_t(x1), f_t(x2), ... , f_t(x2N)]; [2*batch, t_dim]
        
        N = f_s.size(0) // 2

        f_s = self.predictor(f_s) # [2N, s_dim] -> [2N, t_dim]
        features = torch.cat([f_s, f_t], dim=0) # [4N, t_dim]

        anchor = f_s
        contrast = features

        contrast_T = contrast.transpose(0, 1) # [t_dim, 4N]
        
        product = torch.mm(anchor, contrast_T) # [2N, 4N]

        anchor_norm = torch.norm(anchor, dim=-1).unsqueeze(-1) # [2N, 1]
        contrast_norm = torch.norm(contrast, dim=-1).unsqueeze(-1) # [4N, 1]
        norm_matrix = torch.mm(anchor_norm, contrast_norm.transpose(0, 1)) # [2N, 1] * [1, 4N] -> [2N, 4N]
        
        sim_matrix = product / norm_matrix
        sim_matrix = sim_matrix / self.temperature
        sim_matrix = torch.exp(sim_matrix)

        pos_block1 = torch.zeros(N, N) # [N, N]
        pos_block2 = torch.eyes(N) # [N, N]
        
        pos_block3 = torch.cat([pos_block1, pos_block2], dim=0) # [2N, N]
        pos_block4 = torch.cat([pos_block2, pos_block1], dim=0) # [2N, N]
        
        pos_block5 = torch.cat([pos_block3, pos_block4], dim=1) # [2N, 2N]
        pos_block6 = block2.repeat(2, 2) # [2N, 2N]
        pos_mask = torch.cat([pos_block5, pos_block6], dim=1) # [2N, 4N]

        neg_block1 = torch.ones(2*N, 2*N) - torch.eyes(2*N) # [2N, 2N]
        neg_block2 = torch.ones(2*N, 2*N) # [2N, 2N]
        neg_mask = torch.cat([neg_block1, neg_block2], dim=1) # [2N, 4N]

        numerator = sim_matrix * pos_mask
        denominator = sim_matrix * neg_mask

        numerator = torch.sum(numerator, dim=-1) # [2N, 1]
        denominator = torch.sum(denominator, dim=-1) # [2N, 1]

        l = -torch.log(numerator / denominator) # [2N, 1]

        loss = l.mean()

        return loss

class SimclrDistillv3(nn.Module):
    """ OURS : SimCLR-base distillation v3"""
    def __init__(self, opt):
        super(SimlrDistillv1, self).__init__()
        self.predictor = prediction_MLP(in_dim=opt.s_dim,
                            hidden_dim=opt.h_dim, out_dim=opt.t_dim)
        
        self.temperature = 1.0

    def forward(self, f_s, f_t):
        # batch = [x1, ... ,xN] --aug--> [x1, x2, ... ,x2N-1 x2N]
        # x1 -> x1, xN+1 / xk -> xk, xN+k
        # f_s = [f_s(x1), f_s(x2), ... , f_s(x2N)]; [2*batch, s_dim]
        # f_t = [f_t(x1), f_t(x2), ... , f_t(x2N)]; [2*batch, t_dim]
        
        N = f_s.size(0) // 2

        f_s = self.predictor(f_s) # [2N, s_dim] -> [2N, t_dim]
        features = torch.cat([f_s, f_t], dim=0) # [4N, t_dim]

        anchor = f_s
        contrast = features

        contrast_T = contrast.transpose(0, 1) # [t_dim, 4N]
        
        product = torch.mm(anchor, contrast_T) # [2N, 4N]

        anchor_norm = torch.norm(anchor, dim=-1).unsqueeze(-1) # [2N, 1]
        contrast_norm = torch.norm(contrast, dim=-1).unsqueeze(-1) # [4N, 1]
        norm_matrix = torch.mm(anchor_norm, contrast_norm.transpose(0, 1)) # [2N, 1] * [1, 4N] -> [2N, 4N]
        
        sim_matrix = product / norm_matrix
        sim_matrix = sim_matrix / self.temperature
        sim_matrix = torch.exp(sim_matrix)

        pos_block1 = torch.zeros(N, N) # [N, N]
        pos_block2 = torch.eyes(N) # [N, N]
        
        pos_block3 = torch.cat([pos_block1, pos_block2], dim=0) # [2N, N]
        pos_block4 = torch.cat([pos_block2, pos_block1], dim=0) # [2N, N]
        
        pos_block5 = torch.cat([pos_block3, pos_block4], dim=1) # [2N, 2N]
        pos_block6 = block2.repeat(2, 2) # [2N, 2N]
        pos_mask = torch.cat([pos_block5, pos_block6], dim=1) # [2N, 4N]

        neg_block1 = torch.ones(2*N, 2*N) - torch.eyes(2*N) # [2N, 2N]
        neg_block2 = pos_block6
        neg_mask = torch.cat([neg_block1, neg_block2], dim=1) # [2N, 4N]

        numerator = sim_matrix * pos_mask
        denominator = sim_matrix * neg_mask

        numerator = torch.sum(numerator, dim=-1) # [2N, 1]
        denominator = torch.sum(denominator, dim=-1) # [2N, 1]

        l = -torch.log(numerator / denominator) # [2N, 1]

        loss = l.mean()

        return loss

class SimclrDistillv4(nn.Module):
    """ OURS : SimCLR-base distillation v4"""
    def __init__(self, opt):
        super(SimlrDistillv1, self).__init__()
        self.predictor = prediction_MLP(in_dim=opt.s_dim,
                            hidden_dim=opt.h_dim, out_dim=opt.t_dim)
        
        self.temperature = 1.0

    def forward(self, f_s, f_t):
        # batch = [x1, ... ,xN] --aug--> [x1, x2, ... ,x2N-1 x2N]
        # x1 -> x1, xN+1 / xk -> xk, xN+k
        # f_s = [f_s(x1), f_s(x2), ... , f_s(x2N)]; [2*batch, s_dim]
        # f_t = [f_t(x1), f_t(x2), ... , f_t(x2N)]; [2*batch, t_dim]
        
        N = f_s.size(0) // 2

        f_s = self.predictor(f_s) # [2N, s_dim] -> [2N, t_dim]

        anchor = f_s
        contrast = f_t

        contrast_T = contrast.transpose(0, 1) # [t_dim, 2N]
        
        product = torch.mm(anchor, contrast_T) # [2N, 2N]

        anchor_norm = torch.norm(anchor, dim=-1).unsqueeze(-1) # [2N, 1]
        contrast_norm = torch.norm(contrast, dim=-1).unsqueeze(-1) # [2N, 1]
        norm_matrix = torch.mm(anchor_norm, contrast_norm.transpose(0, 1)) # [2N, 1] * [1, 2N] -> [2N, 2N]
        
        sim_matrix = product / norm_matrix
        sim_matrix = sim_matrix / self.temperature
        sim_matrix = torch.exp(sim_matrix)

        pos_mask = (torch.eyes(N)).repeat(2, 2) # [2N, 2N]
        neg_mask = torch.ones(2*N, 2*N) # [2N, 2N]

        numerator = sim_matrix * pos_mask
        denominator = sim_matrix * neg_mask

        numerator = torch.sum(numerator, dim=-1) # [2N, 1]
        denominator = torch.sum(denominator, dim=-1) # [2N, 1]

        l = -torch.log(numerator / denominator) # [2N, 1]

        loss = l.mean()

        return loss
