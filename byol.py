import os, sys, argparse, time, math
import torch
import torch.nn as nn

from dataset.cifar import CIFAR10Instance, CIFAR100Instance

from helper.pretrain import init
from util import TwoCropTransform, TwoCropTransformWithBase
from util import save_model, AverageMeter, BYOL

from models import model_dict
from optimizer import LR_Scheduler

from torchvision import transforms, datasets
from tensorboardX import SummaryWriter

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=100,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=300,
                        help='number of training epochs')
    parser.add_argument('--init_epochs', type=int, default=30,
                        help='init training for two-stage methods')

    # optimizer
    parser.add_argument('--lr', type=float, default=0.03,
                        help='learning rate')
    parser.add_argument('--wd', type=float, default=5e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum for SGD')

    # model / dataset
    parser.add_argument('--model', type=str, default='ResNet50')
    parser.add_argument('--dataset', type=str, default='cifar100',
                        choices=['cifar10', 'cifar100', 'path'])

    # custom dataset
    parser.add_argument('--mean', type=str, help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str, help='std of dataset in path in form of str tuple')
    parser.add_argument('--data_folder', type=str, default=None,
                        help='path to custom dataset')
    parser.add_argument('--img_size', type=int, default=32,
                        help='parameter for RandomResizedCrop')
    
    # distillation
    parser.add_argument('--distill', type=str, default='OURS',
                        help='distillation method')
    parser.add_argument('--teacher', type=str, default='teacher.pth',
                        help='path to teacher')
    parser.add_argument('--beta', type=float, default=1.,
                        help='weight for distillation loss term')
    
    # NCE distillation
    parser.add_argument('--feat_dim', default=128, type=int,
                        help='feature dimension')
    parser.add_argument('--mode', default='exact', type=str,
                        choices=['exact', 'relax'], help='mode for CRD')
    parser.add_argument('--nce_k', default=16384, type=int,
                        help='number of negative samples for NCE')
    parser.add_argument('--nce_t', default=0.07, type=float,
                        help='temperature parameter for softmax')
    parser.add_argument('--nce_m', default=0.5, type=float,
                        help='momentum for non-parametric updates')
    
    # hint layer
    parser.add_argument('--hint_layer', default=2, type=int,
                        choices=[0, 1, 2, 3, 4], help='hint layer for FitNet')

    # OURS
    parser.add_argument('--weighted', action='store_true',
                        help='weighted loss')
    parser.add_argument('--weighting', type=str, default='naive',
                        choices=['naive', 'l1_normalized', 'l2_normalized', 'softmax'], help='weighting method')

    # SimCLR Loss
    parser.add_argument('--temp', type=float, default=0.5,
                        help='temperature for SimCLR loss function')

    # others
    parser.add_argument('--memo', type=str, default='',
                        help='additional memo for ckpt path')

    opt = parser.parse_args()

    # set different learning rate from these 4 models
    if opt.model in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2']:
        opt.lr = 0.01
    
    # check if dataset is path that passed required arguments
    if opt.dataset == 'path':
        assert opt.data_folder is not None \
            and opt.mean is not None \
            and opt.std is not None

    # set the path according to the environment
    opt.model_path = './save/BYOL/{}_models'.format(opt.dataset)
    opt.tb_path = './save/BYOL/{}_tensorboards'.format(opt.dataset)

    opt.model_name = '{}_{}_lr_{}_decay_{}_bsz_{}_beta_{}{}'.\
        format(opt.dataset, opt.model,
            opt.lr, opt.wd, opt.batch_size, opt.beta, opt.memo)

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)
    
    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)
    
    if opt.data_folder is None:
        opt.data_folder = './data'
    
    # device setting
    opt.device = torch.device('cpu')
    if torch.cuda.is_available():
        opt.device = torch.device('cuda')

    return opt

def set_loader(opt):

    # construct data loader
    if opt.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif opt.dataset == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif opt.dataset == 'path':
        mean = eval(opt.mean)
        std = eval(opt.std)
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))

    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize])

    if opt.dataset == 'cifar10':
        train_dataset = CIFAR10Instance(root=opt.data_folder,
                        transform=TwoCropTransform(train_transform),
                        download=True)
    elif opt.dataset == 'cifar100':
        train_dataset = CIFAR100Instance(root=opt.data_folder,
                        transform=TwoCropTransform(train_transform),
                        download=True)
    elif opt.dataset == 'path':
        train_dataset = datasets.ImageFolder(root=opt.data_folder,
                        transform=TwoCropTransform(train_transform))
    else:
        raise ValueError(opt.dataset)
    
    n_data = len(train_dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size,
        num_workers=opt.num_workers, pin_memory=True)

    return train_loader, n_data

def set_model(opt, train_loader, n_data):
    if opt.model == 'ResNet50':
        opt.s_dim = 2048
    elif opt.model == 'ResNet18':
        opt.s_dim = 512
    
    model = BYOL(opt, model_dict[opt.model])
    model = model.to(opt.device)
    return model

def set_optimizer(opt, model, num_iter):
    optimizer = torch.optim.SGD(model.parameters(),
                 lr=opt.lr, momentum=opt.momentum, weight_decay=opt.wd)

    # cosine decay scheduler + warmup
    lr_scheduler = LR_Scheduler(
        optimizer,
        warmup_epochs=10, warmup_lr=0, 
        num_epochs=800, base_lr=opt.lr,
        final_lr=0, 
        iter_per_epoch=num_iter, # len(train_loader)
    )

    return optimizer, lr_scheduler

def train(train_loader, model, optimizer, lr_scheduler, epoch, opt):
    """ one epoch training """

    model.train()

    losses_byol = AverageMeter()

    end = time.time()
    for idx, ((x1, x2), _, index) in enumerate(train_loader):
        x1, x2 = x1.to(opt.device), x2.to(opt.device)

        # ===================forward=====================
        loss_byol = model((x1, x2))

        loss = loss_byol
        losses_byol.update(loss_byol.item(), len(index))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.step(epoch / opt.epochs)
        lr_scheduler.step()

        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}/{1}][{2}/{3}]'
                  ' BYOL loss {loss_byol.val:.6f} ({loss_byol.avg:.6f})'
                  ' | lr {lr:.6f} | time :{time:.2f}s'.format(
                   epoch, opt.epochs, idx + 1, len(train_loader),
                   time=time.time() - end, loss_byol=losses_byol,
                   lr=lr_scheduler.get_lr()))
            
            end = time.time()
            sys.stdout.flush()
        
        #torch.cuda.empty_cache()

    return losses_byol.avg

def main():
    opt = parse_option()
    train_loader, n_data = set_loader(opt)
    model = set_model(opt, train_loader, n_data)
    optimizer, lr_scheduler = set_optimizer(opt, model, len(train_loader))

    writer = SummaryWriter(logdir=opt.tb_folder)

    for epoch in range(1, opt.epochs+1):
        end = time.time()
        loss_byol = train(train_loader, model, optimizer, lr_scheduler, epoch, opt)
        print('epoch {}, total time {:.2f}s'.format(epoch, time.time()-end))
        
        writer.add_scalar('train loss', loss_byol, epoch)
        writer.add_scalar('learning_rate', lr_scheduler.get_lr(), epoch)

        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model.model, optimizer, opt, epoch, save_file)

    # save the last model
    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    save_model(model.model, optimizer, opt, opt.epochs, save_file)

if __name__ == '__main__':
    main()
