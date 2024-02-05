import argparse
import os.path

import matplotlib.pyplot as plt

from utils.dataloder import *
import dynconv
import torch
import models
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
from torch.nn.parallel import DistributedDataParallel as DDP

cudnn.benchmark = True
device='cuda'



## CRITERION


def prepare_for_training(args):
    start_epoch = -1
    best_prec1 = 0
    
    if not args.evaluate and len(args.save_dir) > 0:
        if not os.path.exists(os.path.join(args.save_dir)):
            os.makedirs(os.path.join(args.save_dir))
            
    # optionally resume from a checkpoint
    if args.resume is not None:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint["epoch"] - 1
            best_prec1 = checkpoint["best_prec1"]
            model_state = checkpoint["state_dict"]
            optimizer_state = checkpoint["optimizer"]
            logger.info(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]
                )
            )
            if start_epoch >= args.epochs:
                logger.info(
                    "Launched training for {}, checkpoint already run {}".format(args.epochs,start_epoch)
                )
                exit(1)
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))
            model_state = None
            optimizer_state = None
    else:
        model_state = None
        optimizer_state = None

    class Loss(nn.Module):
        def __init__(self):
            super(Loss, self).__init__()
            self.task_loss = nn.CrossEntropyLoss().to(device=device)
            # if args.loss == "CE":
            #     self.task_loss = nn.CrossEntropyLoss().to(device=device)
            # elif args.loss == "LS":
            #     self.task_loss = dynconv.LabelSmoothing().to(device=device)
            # else:
            #     raise NotImplementedError
            if "mix" in args.sparsity_type:
                sparsity_type = "{}{}".format(args.sparsity_type,args.mix_number)
            logger.info(sparsity_type)
            self.sparsity_loss = dynconv.SparsityCriterion(args.budget, args.epochs,type = sparsity_type) if args.budget >= 0 else None
    
        def forward(self, output, target, meta):
            l = self.task_loss(output, target)
            recoder.add('loss_task', l.item())
            if self.sparsity_loss is not None:
                l += 10 * self.sparsity_loss(meta)
            return l
        
    criterion = Loss()

    # Pytorch Dataloader
    train_loader, train_loader_len = get_pytorch_train_loader(
        args.train_data_path,
        224,
        args.train_batch_size,
        1000,
        False,
        workers=args.workers
    )

    val_loader, val_loader_len = get_pytorch_val_loader(
        args.val_data_path,
        224,
        args.test_batch_size,
        1000,
        False
    )

    ## MODEL
    net_module = models.__dict__[args.model]
    model = net_module(sparse=args.budget >= 0,
                       pretrained=args.pretrained,
                       type=args.sparsity_type,
                       use_ca=args.use_ca).to(device=device)

    if args.pretrained is not None and args.resume is None:
        print("load pretrained model from {}".format(args.pretrained))
        pretrained_model =  torch.load(args.pretrained)
        model.load_state_dict(pretrained_model,strict=False)                       
    

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=args.lr_decay,
                                                        last_epoch=start_epoch)
    start_epoch += 1
    if args.distributed:
        model = DDP(model, device_ids=[args.gpu_id], output_device=args.gpu_id,find_unused_parameters=True)
    
    if model_state is not None:
        model.load_state_dict(model_state)
    
    return (model,criterion,optimizer,lr_scheduler,train_loader,
            val_loader,train_loader_len,val_loader_len,start_epoch,best_prec1)

def main():
    args = BaseConfig
    logger.info('Args:{}'.format(args))


    model, criterion, optimizer, lr_scheduler, train_loader, \
    val_loader, train_loader_len,val_loader_len, start_epoch, best_prec1 = prepare_for_training(args)

    # show_instance(val_loader)
    
    ## Count number of params
    logger.info("* Number of trainable parameters:{}".format(utils.count_parameters(model)))
    ## EVALUATION
    if args.evaluate:
        # evaluate on validation set
        logger.info("########## Evaluation ##########")
        # prec1 = validate(args, val_loader, model, val_loader_len,criterion, start_epoch)
        save_flops_distribution(args, val_loader, model, val_loader_len,criterion, start_epoch)
        return
    ## TRAINING
    for epoch in range(start_epoch, args.epochs):
        logger.info("########## Epoch {} ##########".format(epoch))

        # train for one epoch
        logger.info('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        train(args, train_loader,train_loader_len, model, criterion, optimizer, epoch)
        lr_scheduler.step()

        # evaluate on validation set
        prec1 = validate(args, val_loader, model,val_loader_len, criterion, epoch)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        if args.local_rank == 0:
            utils.save_checkpoint({
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch + 1,
                'best_prec1': best_prec1,
            }, folder=args.save_dir, is_best=is_best)

        logger.info(" * Best prec1: {}".format(best_prec1))

def train(args, train_loader,train_loader_len, model, criterion, optimizer, epoch):
    """
    Run one train epoch
    """
    model.train()

    if epoch < args.lr_decay[0]:
        gumbel_temp = 5.0
    elif epoch < args.lr_decay[1]:
        gumbel_temp = 2.5
    else:
        gumbel_temp = 1
    gumbel_noise = False if epoch > 0.8 * args.epochs else True

    for input, target in tqdm.tqdm(train_loader, total=train_loader_len, ascii=True, mininterval=5):
        
        # compute output
        meta = {'masks': [], 'device': device, 'gumbel_temp': gumbel_temp, 'gumbel_noise': gumbel_noise, 'epoch': epoch}
        output, meta = model(input, meta)
        loss = criterion(output, target, meta)
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        recoder.tick()

@torch.no_grad()
def validate(args, val_loader, model,val_loder_len, criterion, epoch):
    """
    Run evaluation
    """
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()

    # switch to evaluate mode
    model = flopscounter.add_flops_counting_methods(model)
    model.eval()
    model.start_flops_count()
    model.reset_flops_count()

    for input, target in tqdm.tqdm(val_loader, total=val_loder_len, ascii=True, mininterval=5):

        # compute output
        meta = {'masks': [], 'device': device, 'gumbel_temp': 1.0, 'gumbel_noise': False, 'epoch': epoch}
        output, meta = model(input, meta)
        output = output.float()

        # measure accuracy and record loss
        prec1,prec5 = utils.accuracy(output.data, target, topk=(1,5))
        if torch.distributed.is_initialized():
            prec1 = utils.reduce_tensor(prec1)
            prec5 = utils.reduce_tensor(prec5)
        

        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))
        
        if args.plot_ponder:
            viz.plot_image(input)
            viz.plot_ponder_cost(meta['masks'])
            viz.plot_masks(meta['masks'])
            viz.showKey()

    logger.info('* Epoch {} - Prec@1 {:.3f} - Prec@5 {:.3f}'.format(epoch,top1.avg,top5.avg))
    logger.info('* average FLOPS (multiply-accumulates, MACs) per image:  {:.6f} MMac'.
                format(model.compute_average_flops_cost()[0]/1e6))
    model.stop_flops_count()
    return top1.avg


@torch.no_grad()
def save_flops_distribution(args, val_loader, model,val_loder_len, criterion, epoch):
    """
    Run evaluation
    """
    import numpy
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()

    # switch to evaluate mode
    model = flopscounter.add_flops_counting_methods(model)
    model.eval()
    model.start_flops_count()
    model.reset_flops_count()

    flops_distribution = []

    for input, target in tqdm.tqdm(val_loader, total=val_loder_len, ascii=True, mininterval=5):
        assert input.size()[0] == 1
        # compute output
        meta = {'masks': [], 'device': device, 'gumbel_temp': 1.0, 'gumbel_noise': False, 'epoch': epoch}
        output, meta = model(input, meta)
        flops_distribution.append(model.compute_signle_flops_cost()*1e-6)
    model.stop_flops_count()
    data = numpy.array(flops_distribution)
    numpy.save(os.path.join(args.save_dir,'test.npy'),data)
    print("save flops distribution")



def save_image_tensor2pillow(input_tensor: torch.Tensor, filename):
    """
    将tensor保存为pillow
    :param input_tensor: 要保存的tensor
    :param filename: 保存的文件名
    """
    from PIL import Image

    def unnormalize(tensor,mean,std):
        if mean.ndim == 1:
            mean = mean.view(1,-1, 1, 1)
        if std.ndim == 1:
            std = std.view(1,-1, 1, 1)
        tensor.mul_(std).add_(mean)
        return tensor

    
    assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
    # 复制一份
    input_tensor = input_tensor.clone().detach()
    # 到cpu
    input_tensor = input_tensor.to(torch.device('cpu'))
    # 反归一化
    MEAN = torch.FloatTensor([0.485, 0.456, 0.406])
    STD = torch.FloatTensor([0.229, 0.224, 0.225])
    input_tensor = unnormalize(input_tensor,MEAN,STD)
    # 去掉批次维度
    input_tensor = input_tensor.squeeze()
    # 从[0,1]转化为[0,255]，再从CHW转为HWC，最后转为numpy
    input_tensor = input_tensor.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
    # 转成pillow
    im = Image.fromarray(input_tensor)
    filename = os.path.join(BaseConfig.save_dir,filename)
    im.save(filename)
    
def show_instance(val_loader):
    import numpy
    data = np.load("./imagenet_exp/show_flops_distribution/test.npy")
    idx = np.argsort(data)
    idx1 = idx[:20]
    idx2 = idx[10000:10020]
    idx3 = idx[20000:20020]
    idx4 = idx[30000:30020]
    idx5 = idx[40000:40020]
    idx6 = idx[-20:]
    idx = np.concatenate((idx1,idx2,idx3,idx4,idx5,idx6))
    
    idx_set = [ (idx[i],i) for i in range(idx.shape[0])]
    idx_set = sorted(idx_set, key=lambda x:x[0])
    print(idx)
    print(data[idx])
    print(idx_set)
    exit(0)
    print("The number of img need to save: {}".format(len(idx_set)))
    j = 0
    for i , (input, target) in enumerate(val_loader):
        if i == idx_set[j][0]:
            # save img 
            save_image_tensor2pillow(input,"{}.jpg".format(idx_set[j][1]))
            j+=1
    # print(idx_set)
    exit()

if __name__ == "__main__":
    main()    
