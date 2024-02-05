# import torch
# # import models
# # from torchvision import models



# class Wapper:
#     def __init__(self,index):
#         self.index = index
#     def __enter__(self):
#         print("construct a Wapper-{}".format(self.index))
    
#     def __exit__(self, type, value, traceback):
#         print("deconstruct a Wapper-{}".format(self.index))


# def isContiguous(tensor):
    

#     z = 1
#     d = tensor.dim() - 1
#     size = tensor.size()
#     stride = tensor.stride()
#     print("stride={} size={}".format(stride, size))
#     while d >= 0:
#         if size[d] != 1:
#             if stride[d] == z:
#                 print("dim {} stride is {}, next stride should be {} x {}".format(d, stride[d], z, size[d]))
#                 z *= size[d]                
#             else:
#                 print("dim {} is not contiguous. stride is {}, but expected {}".format(d, stride[d], z))
#                 return False
#         d -= 1
#     return True

# if __name__ == '__main__':
    
#     # a = torch.randn(4,3,2,2)
#     # mask = torch.LongTensor([1,0,0,1]).view(-1,1,1,1).bool()
#     # prev_mask = torch.LongTensor([0,0,1]).view(1,-1,1,1).bool()
#     # print(a)
#     # a = torch.masked_select(a,mask).view(-1,*a.size()[1:])
#     # a = torch.masked_select(a,prev_mask).view(a.size(0),-1,*a.size()[2:])
#     # print(a)
#     # print(a.size())
#     # s_mask = torch.nonzero(a[0,0,:,:]>0)
#     # print(torch.nonzero(prev_mask.squeeze())[0])
#     # print(s_mask[0,:])
#     # print(s_mask[1,:])

#     # a = torch.randn(1,1,5,5)
#     # b = torch.randn(1,5,1,1)
#     # s_mask = a > 0;
#     # c_mask = b > 0;
#     # c_mask_idx = torch.nonzero(c_mask.squeeze()).squeeze()
#     # s_mask_idx = torch.nonzero(s_mask.squeeze())
#     # s_mask_h_idx = s_mask_idx[:, 0].contiguous()
#     # s_mask_w_idx = s_mask_idx[:, 1].contiguous()
#     # print(a)
#     # print(b)
#     # print(c_mask_idx)
#     # print(s_mask_h_idx)
#     # print(s_mask_w_idx)
#     # a = torch.nn.Conv2d(3,5,3,1,bias=True)
#     # a = torch.masked_select(a.bias.data,c_mask.squeeze())
#     # print(a)

#     # a = torch.nn.Linear(5,3)
#     # b = torch.randn(2,5)
#     # print(a.weight.size())
#     # print(a(b).size())
#     # print(s_mask_idx.size(0))


#     # a = torch.LongTensor([1,0,1])

#     # b = torch.LongTensor([[1,0,1],
#     #                       [1,1,0],
#     #                       [1,0,1]])
#     # c_mask_idx = torch.nonzero(a).squeeze();
#     # s_mask_idx = torch.nonzero(b).squeeze();
#     # s_mask_h_idx = s_mask_idx[:, 0].contiguous();
#     # s_mask_w_idx = s_mask_idx[:, 1].contiguous()

#     # print(c_mask_idx)
#     # print(s_mask_idx)
#     # print(s_mask_h_idx)
#     # print(s_mask_w_idx)

#     # mask_cnt = s_mask_h_idx.size(0);
#     # out_channel = c_mask_idx.size(0);
#     # height = b.size(0)
#     # width = b.size(1)
#     # output = torch.zeros(a.size(0),*b.size()).view(-1)
#     # c = torch.randn(c_mask_idx.size(0),s_mask_idx.size(0));
#     # c_view = c.view(-1)
#     # output_size =  mask_cnt * out_channel;

#     # for i in range (output_size):
#     #     m_index = i % mask_cnt;
#     #     img_h = s_mask_h_idx[m_index]
#     #     img_w = s_mask_w_idx[m_index]
#     #     c_index = int(i/mask_cnt)
#     #     img_c = c_mask_idx[c_index]
#     #     output[(img_c* height + img_h) * width + img_w] = c_view[i]
#     # print(c)
#     # print(output.view(a.size(0),*b.size()))

#     def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
#         """3x3 convolution with padding"""
#         return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
#                         padding=dilation, groups=groups, bias=False, dilation=dilation)

#     from torch import nn
#     from sparse_conv2d import SparseConv2d

#     # class BasicBlock(nn.Module):
#     #     def __init__(inplanes, planes, stride=1):
#     #         self.conv1 = conv3x3(inplanes, planes, stride)
#     #         self.relu = nn.ReLU(inplace=True)
#     #         self.conv2 = conv3x3(planes, planes)

#     #     def forward(self,input):
#     #         out = self.conv1(input)
#     #         out = self.relu(out)
#     #         out = self.conv2(out)
#     #         out = out + identity
#     #         return self.relu(out)

#     # class SparseConv2dBasicBlock(nn.Module):

#     #     def __init__(inplanes, planes, stride=1):
#     #         self.conv1 = SparseConv2d(inplanes, planes,3,stride,1,bias=False,last_conv = False,activation = self.relu)
#     #         self.relu = nn.ReLU(inplace=True)
#     #         self.conv2 = SparseConv2d(planes, planes, 3, 1, 1, bias=False,last_conv = True)

#     #     
#     #     def forward(self,input):
#     #         out = self.conv1(input)
#     #         out = self.conv2(out)
#     #         out = out + identity
#     #         return self.relu(out)            


#     print("-------------------cpu implement--------------------")
#     c_mask1 = torch.LongTensor([1,0,1]).view(1,3,1,1).bool()
#     s_mask1 = torch.LongTensor([[1,0,1],
#                                 [1,1,0],
#                                 [1,0,1]]).view(1,1,3,3).bool()
#     x = torch.randn(1,4,5,5)
    
#     # relu = nn.ReLU(inplace = True)
#     relu = None
#     conv1 = conv3x3(4, 3, 2)
#     conv2 = SparseConv2d(3, 4, 3, 2, 1, bias=False, 
#                         last_conv = True, activation = relu)
#     conv2.weight.data = conv1.weight.data.clone()
#     mask_dict = {"prev_c_mask":None,
#                  "c_mask":c_mask1,
#                  "s_mask":s_mask1}
#     print(conv2(x,mask_dict).squeeze())
#     print((conv1(x)*s_mask1.float()*c_mask1.float()).squeeze())
#     # print(relu((conv1(x)*s_mask1.float()*c_mask1.float()).squeeze()))
#     print("-------------------cuda implement--------------------")
#     c_mask1 = c_mask1.cuda()
#     s_mask1 = s_mask1.cuda()
#     x = x.cuda()
    
#     relu = nn.ReLU(inplace = True)
#     conv1 = conv1.cuda()
#     conv2 = conv2.cuda()
#     conv2.weight.data = conv1.weight.data.clone()
#     mask_dict = {"prev_c_mask":None,
#                  "c_mask":c_mask1,
#                  "s_mask":s_mask1}
#     print(conv2(x,mask_dict).squeeze())
#     print((conv1(x)*s_mask1.float()*c_mask1.float()).squeeze())
#     # print(relu((conv1(x)*s_mask1.float()*c_mask1.float()).squeeze()))
#     # print(torch.get_num_threads())
    
#     # print(torch.randn(1).is_cuda())
#     conv3 = torch.nn.Conv2d(3,3,3,groups = 3)
#     print(conv3.weight.size())


from utils.dataloder import *
import models_test
import inference_model
from utils.config import BaseConfig
from utils.utils import fuse_model
from utils.logger import logger
from copy import deepcopy
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import tqdm
import utils.utils as utils
from copy import deepcopy
import os
import time

device = 'cuda'

def main():
    args = BaseConfig
    net_module = models_test.__dict__[args.model]
    model = net_module(sparse=args.budget >= 0,type = args.sparsity_type,use_ca = args.use_ca).eval()
    ## CHECKPOINT
    start_epoch = -1
    best_prec1 = 0

    if args.resume:
        resume_path = args.resume
        if not os.path.isfile(resume_path):
            resume_path = os.path.join(resume_path, 'checkpoint.pth')
        if os.path.isfile(resume_path):
            print("=> loading checkpoint '{}'".format(resume_path))
            checkpoint = torch.load(resume_path)
            start_epoch = checkpoint['epoch']-1
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}'' (epoch {}, best prec1 {})"
                        .format(resume_path,checkpoint['epoch'],checkpoint['best_prec1']))
        else:
            msg = "=> no checkpoint found at '{}'".format(resume_path)
            if args.evaluate:
                raise ValueError(msg)
            else:
                print(msg)

    new_model = fuse_model(deepcopy(model))

    sparse_module = inference_model.__dict__["sparse_{}".format(args.model)]
    sparse_model = sparse_module(use_ca = args.use_ca).eval()
    sparse_model.load_state_dict(new_model.state_dict())
    # for k , v in model.state_dict().items():
    #     v1 =  sparse_model.state_dict()[k]
    #     print(k, (v!=v1).sum())

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


    sparse_model = sparse_model.cuda()
    model = model.cuda()
    # print(sparse_model)

    if args.inference:
        validate(args,val_loader,sparse_model,val_loader_len,None,0)
    else:
        validate(args,val_loader,model,val_loader_len, None, 0)



@torch.no_grad()
def validate(args, val_loader, model, val_loder_len, criterion, epoch):
    """
    Run evaluation
    """
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    inference_time = utils.AverageMeter()
    model.eval()

    for input, target in tqdm.tqdm(val_loader, total=val_loder_len, ascii=True, mininterval=5):
        meta = {'masks': [], 'device': device, 'gumbel_temp': 1.0, 'gumbel_noise': False, 'epoch': epoch}
        start = time.time()
        output, meta = model(input, meta)
        end = time.time()
        output = output.float()
        # measure accuracy and record loss
        prec1, prec5 = utils.accuracy(output.data, target, topk=(1,5))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))
        inference_time.update(end-start,input.size(0))

    logger.info("inference time = {}".format(inference_time.avg))
    logger.info('* Epoch {} - Prec@1 {:.3f} - Prec@5 {:.3f}'.format(epoch,top1.avg,top5.avg))
    

if __name__ == '__main__':
    main()


# device = 'cpu'

# def main():
#     args = BaseConfig
#     net_module = models_test.__dict__[args.model]
#     sparse_module = inference_model.__dict__["sparse_{}".format(args.model)]
#     model = net_module(sparse=args.budget >= 0,type = args.sparsity_type,use_ca = args.use_ca).eval()
#     model = fuse_model(model)
#     sparse_model = sparse_module(use_ca = args.use_ca).eval()
#     sparse_model.load_state_dict(model.state_dict())

#     mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
#     transform_train = transforms.Compose([
#         transforms.RandomCrop(32, padding=4),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize(mean, std),
#     ])
#     transform_test = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize(mean, std),
#     ])

#     ## DATA
#     trainset = datasets.CIFAR10(root='./data/cifar10', train=True, download=True, transform=transform_train)
#     train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batchsize, shuffle=True, num_workers=args.workers, pin_memory=False)

#     valset = datasets.CIFAR10(root='./data/cifar10', train=False, download=True, transform=transform_test)
#     val_loader = torch.utils.data.DataLoader(valset, batch_size=args.batchsize, shuffle=False, num_workers=4, pin_memory=False)

#     sparse_model = sparse_model
#     validate(args,val_loader,sparse_model,None,0)


# @torch.no_grad()
# def validate(args, val_loader, model, criterion, epoch):
#     """
#     Run evaluation
#     """
#     top1 = utils.AverageMeter()
#     model.eval()

#     num_step = len(val_loader)
#     with torch.no_grad():
#         for input, target in tqdm.tqdm(val_loader, total=num_step, ascii=True, mininterval=5):
#             # input = input.to(device=device, non_blocking=True)
#             # target = target.to(device=device, non_blocking=True)

#             # compute output
#             meta = {'masks': [], 'device': device, 'gumbel_temp': 1.0, 'gumbel_noise': False, 'epoch': epoch}
#             output, meta = model(input, meta)
#             output = output.float()

#             # measure accuracy and record loss
#             prec1 = utils.accuracy(output.data, target)[0]
#             top1.update(prec1.item(), input.size(0))

#             if args.plot_ponder:
#                 viz.plot_image(input)
#                 viz.plot_ponder_cost(meta['masks'])
#                 viz.plot_masks(meta['masks'])
#                 plt.show()

#     logger.info('* Epoch {} - Prec@1 {:.3f}'.format(epoch,top1.avg))
#     return top1.avg

# if __name__ == '__main__':
#     main()
