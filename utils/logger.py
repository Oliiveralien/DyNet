# A simple torch style logger
# (C) Wei YANG 2017
from __future__ import absolute_import
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
import logging
import time
import torch
from .config import BaseConfig
import errno

__all__ = ['logger','recoder']

# def get_logger(file_path):
#     """ Make python logger """
#     logger = logging.getLogger('Dynamic Neural Network')
#     log_format = '%(asctime)s | %(message)s'
#     formatter = logging.Formatter(log_format, datefmt='%m/%d %I:%M:%S %p')
#     path = os.path.join(file_path,"log.txt")
#     file_handler = logging.FileHandler()
#     file_handler.setFormatter(formatter)
#     stream_handler = logging.StreamHandler()
#     stream_handler.setFormatter(formatter)
#
#     logger.addHandler(file_handler)
#     logger.addHandler(stream_handler)
#     logger.setLevel(logging.INFO)
#     return logger

def mkdir_p(path):
    '''make dir if not exist'''
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


class LoggerWapper(object):
    '''Save training process to log file with simple plot function.'''
    def __init__(self, path,local_rank):
        self.local_rank = local_rank
        self.logger = self.get_logger(path)

    def get_logger(self,log_path=None,fmt=None):
        formatter = None
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        if fmt is not None:
            log_format = '%(asctime)s | %(message)s'
            formatter = logging.Formatter(log_format, datefmt='%m/%d %I:%M:%S %p')
        if log_path is not None:
            logger_name = os.path.join(log_path, 'log.txt')
            if not os.path.isdir(log_path):
                mkdir_p(log_path)
            # print(logger_name)
            # if not os.path.exists(logger_name):
            #     print("create!!!!!!!!!!!!")
            #     os.system(r"touch {}".format(logger_name))
            handler = logging.FileHandler(logger_name)
            handler.setLevel(logging.INFO)
            if formatter is not None:
                handler.setFormatter(formatter)
            logger.addHandler(handler)
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        if formatter is not None:
            console.setFormatter(formatter)
        logger.addHandler(console)
        return logger

    def info(self,str,all_rank=False):
        if all_rank:
            self.logger.info("rank_{} {}".format(cfg.local_rank,str))
        elif self.local_rank == 0:
            self.logger.info(str)



def get_logger(log_path=None,fmt=None):
    formatter =None
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if fmt is not None:
        log_format = '%(asctime)s | %(message)s'
        formatter = logging.Formatter(log_format, datefmt='%m/%d %I:%M:%S %p')
    if log_path is not None:
        logger_name = os.path.join(log_path,'log.txt')
        if not os.path.isdir(log_path):
            mkdir_p(log_path)
        # print(logger_name)
        # if not os.path.exists(logger_name):
        #     print("create!!!!!!!!!!!!")
        #     os.system(r"touch {}".format(logger_name))
        handler = logging.FileHandler(logger_name)
        handler.setLevel(logging.INFO)
        if formatter is not None:
            handler.setFormatter(formatter)
        logger.addHandler(handler)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    if formatter is not None:
        console.setFormatter(formatter)
    logger.addHandler(console)
    return logger


class Recoder:
    def __init__(self,logger,interval):
        self.logger = logger
        self.record = {}
        self.step = 0
        self.interval = interval
        
    def add(self,name, val):
        self.record[name] = val

    def tick(self):
        if self.step > 0 and self.step % self.interval == 0:
            self.logger.info('-'*150)
            for name in self.record:
                val = self.record[name]
                if val is not None:
                    self.logger.info("$ {}:{}".format(str(name).ljust(20),val))
        self.step += 1
        
        
    def reset(self):
        self.record = {}
        self.step = 0


logger = get_logger(BaseConfig.save_dir)
recoder = Recoder(logger,BaseConfig.interval)
# distributed_logger = LoggerWapper(BaseConfig.save_dir,BaseConfig.local_rank)


