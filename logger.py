import os
import time
import logging
import torch.distributed as dist


def setup_logger(logpth):
    
    # 确保日志目录存在
    os.makedirs(logpth, exist_ok=True)
    
    logfile = 'A2RNet-{}.log'.format(time.strftime('%Y-%m-%d-%H-%M-%S'))
    logfile = os.path.join(logpth, logfile)
        
    FORMAT = '%(levelname)s %(filename)s(%(lineno)d): %(message)s'
    log_level = logging.INFO
    if dist.is_initialized() and not dist.get_rank()==0:
        log_level = logging.ERROR
    logging.basicConfig(level=log_level, format=FORMAT, filename=logfile)
    logging.root.addHandler(logging.StreamHandler())