import argparse
import os
from functools import partial
import sys


def get_parser(name):
    """ make default formatted parser """
    parser = argparse.ArgumentParser(name, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # print default value always
    parser.add_argument = partial(parser.add_argument, help=' ')
    return parser



class BaseArgs(argparse.Namespace):
    def build_parser(self):
        parser = get_parser("Base config")
        return parser

    def __init__(self):
        self.parser = self.build_parser()
        self.parse_args()
        
        
    def parse_args(self):
        args = self.parser.parse_args()
        super().__init__(**vars(args))
        

parser = argparse.ArgumentParser()
sub_parsers = parser.add_subparsers(help='base')

def config_distributed(args):
    args.distributed = False
    # world_size 相当于进程数,local_rank 相当于进程ID,
    if "WORLD_SIZE" in os.environ:
        # 会启动多个进程，每个进程都有一个local_rank
        args.distributed = int(os.environ["WORLD_SIZE"]) > 1
        args.local_rank = int(os.environ["LOCAL_RANK"])
    else:
        args.local_rank = 0
    args.world_size = 1
    if args.distributed:
        args.gpu_id = args.local_rank % torch.cuda.device_count()
        torch.cuda.set_device(args.gpu_id)
        dist.init_process_group(backend="nccl", init_method="env://")
        args.world_size = torch.distributed.get_world_size()
    return args


def get_imagenet_args():
    global sub_parsers
    imagenet_parser = sub_parsers.add_parser("imagenet")
    imagenet_parser.add_argument('--lr', default=0.025, type=float, help='learning rate')
    imagenet_parser.add_argument('--lr_decay', default=[30, 60, 90], nargs='+', type=int,
                                 help='learning rate decay epochs')
    imagenet_parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    imagenet_parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay')
    imagenet_parser.add_argument('--train_batch_size', default=32, type=int, help='batch size')
    imagenet_parser.add_argument('--test_batch_size', default=32, type=int, help='batch size')
    imagenet_parser.add_argument('--epochs', default=100, type=int, help='number of epochs')
    # arch
    imagenet_parser.add_argument('--model', type=str, default='resnet101', help='network model name')
    imagenet_parser.add_argument('-ca', '--use_ca', default=False, action='store_true', help='evaluation mode')
    imagenet_parser.add_argument('-st', '--sparsity_type', type=str, default='mix',
                                 choices=['spatial', 'channel', 'mix'], help='sparsity type')
    
    imagenet_parser.add_argument('--budget', default=-1, type=float,
                                 help='computational budget (between 0 and 1) (-1 for no sparsity)')
    imagenet_parser.add_argument('-s', '--save_dir', type=str, default='', help='directory to save model')
    imagenet_parser.add_argument('-r', '--resume', default=None, type=str, metavar='PATH',
                                 help='path to latest checkpoint (default: none)')
    imagenet_parser.add_argument('--train_data_path', default='/media/upc/68bafd7e-bd90-4bd2-89a1-35e2323ac508/ImageNet/ILSVRC2012_img_train/train', type=str,
                                 metavar='PATH',
                                 help='ImageNet dataset root')
    imagenet_parser.add_argument('--val_data_path', default='/media/upc/68bafd7e-bd90-4bd2-89a1-35e2323ac508/ImageNet/ILSVRC2012_img_val', type=str,
                                 metavar='PATH',
                                 help='ImageNet dataset root')
    imagenet_parser.add_argument('--upper', default=0.9, type=float)
    imagenet_parser.add_argument('--lower', default=0.1, type=float)
    imagenet_parser.add_argument('-e', '--evaluate', action='store_true', help='evaluation mode')
    imagenet_parser.add_argument('--inference', default=False, action='store_true', help='inference mode')
    imagenet_parser.add_argument('--plot_ponder', action='store_true', help='plot ponder cost')
    imagenet_parser.add_argument('--pretrained', default=None, type=str, metavar='PATH',
                                 help='path to pretrained model (default: none)')
    imagenet_parser.add_argument('--workers', default=8, type=int, help='number of dataloader workers')
    imagenet_parser.add_argument('--show_reduction_ratio', default=False, action='store_true',
                                 help='show reduction ratio')
    imagenet_parser.add_argument('--interval', default=400, type=int, help='print frequence')
    imagenet_parser.add_argument('--send_checkpoint', default=False, action='store_true',
                                 help='send checkpoint to other pc')
    imagenet_parser.add_argument('--mix_number', default=7, type=int, help='mix_number')


def get_cifar_args():
    global sub_parsers
    cifar_parser = sub_parsers.add_parser("cifar")
    cifar_parser.add_argument('-s', '--save_dir', type=str, default=None, help='directory to save model')
    cifar_parser.add_argument('--show_reduction_ratio', default=False, action='store_true', help='show reduction ratio')
    cifar_parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    cifar_parser.add_argument('--lr_decay', default=[150,250], nargs='+', type=int, help='learning rate decay epochs')
    cifar_parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    cifar_parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay')
    cifar_parser.add_argument('--batchsize', default=256, type=int, help='batch size')
    cifar_parser.add_argument('--epochs', default=350, type=int, help='number of epochs')
    cifar_parser.add_argument('--model', type=str, default='resnet32', help='network model name')
    cifar_parser.add_argument('--smooth_coffe', default=0.0, type=float, help='label smooth')

    # parser.add_argument('--resnet_n', default=5, type=int, help='number of layers per resnet stage (5 for Resnet-32)')
    cifar_parser.add_argument('--budget', default=-1, type=float, help='computational budget (between 0 and 1) (-1 for no sparsity)')
    cifar_parser.add_argument('--upper', default=0.9, type=float)
    cifar_parser.add_argument('--lower', default=0.1, type=float)
    cifar_parser.add_argument('-st', '--sparsity_type', type=str, default='spatial',choices=['spatial','channel','mix'], help='sparsity type')
    cifar_parser.add_argument('-r', '--resume', default='', type=str, metavar='PATH',
                  help='path to latest checkpoint (default: none)')
    cifar_parser.add_argument('--pretrained',default=False, action='store_true', help='load pretrained model')
    cifar_parser.add_argument('-ca', '--use_ca',default=False, action='store_true', help='evaluation mode')
    cifar_parser.add_argument('--atten_head',default=False, action='store_true', help='evaluation mode')
    cifar_parser.add_argument('-e', '--evaluate', action='store_true', help='evaluation mode')
    cifar_parser.add_argument('--plot_ponder', action='store_true', help='plot ponder cost')
    cifar_parser.add_argument('--inference',default=False, action='store_true', help='plot ponder cost')
    cifar_parser.add_argument('--workers', default=8, type=int, help='number of dataloader workers')
    cifar_parser.add_argument('--interval', default=200, type=int, help='print frequence')
    cifar_parser.add_argument('--mix_number', default=7, type=int, help='mix_number')
    cifar_parser.add_argument('--loss_alpha', default=10, type=int, help='loss_alpha')
    cifar_parser.add_argument('--gumbel_temp', default=1.0, type=float, help='gumbel_temp')
    cifar_parser.add_argument('--data_path', default='./data/cifar10', type=str, help='data_path')

    # cifar_parser.add_argument('--pretrained',default=False, action='store_true', help='initialize with pretrained model')


def config():
    get_imagenet_args()
    get_cifar_args()
    config, _ = parser.parse_known_args()
    args = config_distributed(config)
    return args

BaseConfig = config()

if __name__ == '__main__':
    BaseConfig.parse_args()
    print(BaseConfig)
    print(BaseConfig.hello)
