from __future__ import division
import os
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIClassificationIterator, DALIGenericIterator
import itertools
import numpy as np
from random import shuffle
import sys



__all__ = ["get_dali_imagenet_dataloader","DataWapper"]


class HybridTrainPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop, dali_cpu=False, local_rank=0, world_size=1):
        super(HybridTrainPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)
        dali_device = "gpu"
        self.input = ops.FileReader(file_root=data_dir, shard_id=local_rank, num_shards=world_size, random_shuffle=True)

        self.decode = ops.ImageDecoderRandomCrop(device="mixed",
                                                 output_type=types.RGB,
                                                 random_aspect_ratio=[0.8, 1.25],
                                                 random_area=[0.1, 1.0],
                                                 num_attempts=100)
        self.resize = ops.Resize(device="gpu", resize_x=224, resize_y=224)
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(224, 224),
                                            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                            std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
        self.coin = ops.CoinFlip(probability=0.5)

    def define_graph(self):
        rng = self.coin()
        self.jpegs, self.labels = self.source()
        images = self.decode(self.jpegs)
        images = self.resize(images)
        output = self.cmnp(images, mirror=rng)
        return [output, self.labels]


class HybridValPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop, size, local_rank=0, world_size=1):
        super(HybridValPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)
        self.input = ops.FileReader(file_root=data_dir, shard_id=local_rank, num_shards=world_size,
                                    random_shuffle=False)
        self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)
        self.res = ops.Resize(device="gpu", resize_shorter=size, interp_type=types.INTERP_TRIANGULAR)
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(224, 224),
                                            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                            std=[0.229 * 255, 0.224 * 255, 0.225 * 255])

    def define_graph(self):
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images)
        return [output, self.labels]


def get_dali_imagenet_dataloader(data_path,logger=None,world_size=1,local_rank=0):
    train_dir = os.path.join(data_path, "train")
    val_dir = os.path.join(data_path, "val")
    pip_train = HybridTrainPipe(batch_size=cfg.data.train_batch_size, num_threads=cfg.data.workers, device_id=local_rank,
                                data_dir=train_dir,crop=224, world_size=world_size, local_rank=local_rank)
    pip_train.build()
    dali_iter_train = DALIClassificationIterator(pip_train, size=pip_train.epoch_size("Reader") // world_size)

    pip_val = HybridValPipe(batch_size=cfg.data.test_batch_size, num_threads=cfg.data.workers, device_id=local_rank,
                            data_dir=val_dir,crop=224, size=256, world_size=world_size, local_rank=local_rank)
    pip_val.build()
    dali_iter_val = DALIClassificationIterator(pip_val, size=pip_val.epoch_size("Reader") // world_size)

    return dali_iter_train,dali_iter_val




# RAII封装 dataloader
# 使用方法
# with DaraWapper(data,use_cuda,dataloder) as wapper:
#       optimizer.zero_grad()
#       outputs = model(wapper.data)
#       loss = criterion(outputs,wapper.label)
#       loss.backward()
#       optimizer.step()
class DataWapper:
    def __init__(self,data,use_cuda,dataloader):
        self.use_dali = isinstance(self.dataloader,DALIClassificationIterator)
        if self.use_dali:
            inputs, targets = data[0]["data"], data[0]["label"].squeeze().long()
        else:
            inputs, targets = data[0], data[1]

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        self.inputs = inputs
        self.targets = targets

    def __enter__(self):
        pass

    @property
    def data(self):
        return self.inputs

    @property
    def label(self):
        return self.targets

    def __exit__(self):
        if self.use_dali
            self.dataloader.reset()

