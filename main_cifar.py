import argparse
import os.path

import matplotlib.pyplot as plt

import dynconv
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import tqdm
import utils.flopscounter as flopscounter
from utils.logger import logger,recoder
from utils.config import BaseConfig
import utils.utils as utils
import utils.viz as viz
from torch.backends import cudnn as cudnn
import models

cudnn.benchmark = True

device = 'cuda'

def load_checkpoint(model,state_dict,pretrained):
    if pretrained:
        model_dict = model.state_dict()
        for k in model_dict.keys():
            if k in state_dict.keys():
                model_dict[k] = state_dict[k]
        model.load_state_dict(model_dict)
    else:
        model.load_state_dict(state_dict)

def main():
    args = BaseConfig
    
    logger.info('Args:{}'.format(args))
    
    
    mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    ## DATA
    trainset = datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batchsize, shuffle=True, num_workers=args.workers, pin_memory=False)

    valset = datasets.CIFAR10(root=args.data_path, train=False, download=True, transform=transform_test)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=args.batchsize, shuffle=False, num_workers=4, pin_memory=False)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    ## MODEL
    net_module = models.__dict__[args.model]
    model = net_module(sparse=args.budget >= 0,type = args.sparsity_type,use_ca = args.use_ca).to(device=device)

    ## CRITERION
    class Loss(nn.Module):
        def __init__(self):
            super(Loss, self).__init__()
            if args.smooth_coffe > 0.0:
                self.task_loss = dynconv.LabelSmoothing(smoothing=args.smooth_coffe)
            else:
                self.task_loss = nn.CrossEntropyLoss().to(device=device)
            if "mix" in args.sparsity_type:
                sparsity_type = "{}{}".format(args.sparsity_type, args.mix_number)
            self.sparsity_loss = dynconv.SparsityCriterion(args.budget, args.epochs,type = sparsity_type) if args.budget >= 0 else None

        def forward(self, output, target, meta):
            l = self.task_loss(output, target) 
            recoder.add('loss_task', l.item())
            if self.sparsity_loss is not None:
                l += args.loss_alpha * self.sparsity_loss(meta)
            return l
    
    criterion = Loss()
    ## OPTIMIZER
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                            momentum=args.momentum,
                            weight_decay=args.weight_decay)

    ## CHECKPOINT
    start_epoch = -1
    best_prec1 = 0



    if args.resume:
        resume_path = args.resume
        if not os.path.isfile(resume_path):
            resume_path = os.path.join(resume_path, 'checkpoint.pth')
        if os.path.isfile(resume_path):
            logger.info("=> loading checkpoint '{}'".format(resume_path))
            checkpoint = torch.load(resume_path)
            if args.pretrained == False:
                start_epoch = checkpoint['epoch']-1
                best_prec1 = checkpoint['best_prec1']
                # model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
            load_checkpoint(model,checkpoint['state_dict'],args.pretrained)
            logger.info("=> loaded checkpoint '{}'' (epoch {}, best prec1 {})"
                        .format(resume_path,checkpoint['epoch'],checkpoint['best_prec1']))
        else:
            msg = "=> no checkpoint found at '{}'".format(resume_path)
            if args.evaluate:
                raise ValueError(msg)
            else:
                logger.info(msg)


    try:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
            milestones=args.lr_decay, last_epoch=start_epoch)
    except:
        logger.info('Warning: Could not reload learning rate scheduler')
    start_epoch += 1
            
    ## Count number of params
    logger.info("* Number of trainable parameters:{}".format(utils.count_parameters(model)))


    ## EVALUATION
    if args.evaluate:
        logger.info("########## Evaluation ##########")
        prec1 = validate(args, val_loader, model, criterion, start_epoch)
        return
        
    ## TRAINING
    for epoch in range(start_epoch, args.epochs):
        logger.info("########## Epoch {} ##########".format(epoch))

        # train for one epoch
        logger.info('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        train(args, train_loader, model, criterion, optimizer, epoch)
        lr_scheduler.step()

        # evaluate on validation set
        prec1 = validate(args, val_loader, model, criterion, epoch)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        utils.save_checkpoint({
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'best_prec1': best_prec1,
        }, folder=args.save_dir, is_best=is_best)

        logger.info(" * Best prec1: {}".format(best_prec1))

def train(args, train_loader, model, criterion, optimizer, epoch):
    """
    Run one train epoch
    """
    model.train()

    # set gumbel temp
    # disable gumbel noise in finetuning stage
    gumbel_temp = args.gumbel_temp
    gumbel_noise = False if epoch > 0.8 * args.epochs else True

    num_step =  len(train_loader)
    for input, target in tqdm.tqdm(train_loader, total=num_step, ascii=True, mininterval=5):

        input = input.to(device=device, non_blocking=True)
        target = target.to(device=device, non_blocking=True)

        # compute output
        meta = {'masks': [], 'device': device, 'gumbel_temp': gumbel_temp, 'gumbel_noise': gumbel_noise, 'epoch': epoch}
        output, meta = model(input, meta)
        loss = criterion(output, target, meta)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        recoder.tick()



def validate(args, val_loader, model, criterion, epoch):
    """
    Run evaluation
    """
    top1 = utils.AverageMeter()

    # switch to evaluate mode
    model = flopscounter.add_flops_counting_methods(model)
    model.eval().start_flops_count()
    model.reset_flops_count()

    num_step = len(val_loader)
    with torch.no_grad():
        for input, target in tqdm.tqdm(val_loader, total=num_step, ascii=True, mininterval=5):
            input = input.to(device=device, non_blocking=True)
            target = target.to(device=device, non_blocking=True)

            # compute output
            meta = {'masks': [], 'device': device, 'gumbel_temp': 1.0, 'gumbel_noise': False, 'epoch': epoch}
            output, meta = model(input, meta)
            output = output.float()

            # measure accuracy and record loss
            prec1 = utils.accuracy(output.data, target)[0]
            top1.update(prec1.item(), input.size(0))

            if args.plot_ponder:
                viz.plot_image(input)
                viz.plot_ponder_cost(meta['masks'])
                viz.plot_masks(meta['masks'])
                plt.show()
    logger.info('* Epoch {} - Prec@1 {:.3f}'.format(epoch,top1.avg))
    logger.info('* average FLOPS (multiply-accumulates, MACs) per image:  {:.6f} MMac'.
                format(model.compute_average_flops_cost()[0]/1e6))
    if BaseConfig.show_reduction_ratio:
        model.show_reduction_ratio()
    model.stop_flops_count()
    return top1.avg

if __name__ == "__main__":
    main()    
