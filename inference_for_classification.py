# Single GPU:
# CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master-port 1215 inference_for_classification.py \
#   --cfg configs/swin/Senatra.yaml --resume nativeseg.pth --data-path /raid/Datasets/imagenet/

import os
import time
import json
import random
import argparse
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from timm.utils import accuracy, AverageMeter

from config import get_config
from models import build_model
from data import build_loader
from logger import create_logger
from utils import reduce_tensor

PYTORCH_MAJOR_VERSION = int(torch.__version__.split('.')[0])


def parse_option():
    parser = argparse.ArgumentParser('Senatra classification inference', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar='FILE', help='path to config file')
    parser.add_argument("--opts", default=None, nargs='+')
    parser.add_argument('--batch-size', type=int, default=64, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, required=True, help='path to ImageNet dataset root')
    parser.add_argument('--resume', type=str, required=True, help='path to checkpoint .pth')
    parser.add_argument('--zip', action='store_true')
    parser.add_argument('--cache-mode', type=str, default='no', choices=['no', 'full', 'part'])
    parser.add_argument('--output', default='output', type=str)
    parser.add_argument('--tag', default='default')
    parser.add_argument('--eval', action='store_true', default=True)
    parser.add_argument('--disable_amp', action='store_true')
    parser.add_argument('--amp-opt-level', type=str, choices=['O0', 'O1', 'O2'])
    parser.add_argument('--accumulation-steps', type=int, default=1)
    parser.add_argument('--use-checkpoint', action='store_true')
    parser.add_argument('--fused_window_process', action='store_true')
    parser.add_argument('--fused_layernorm', action='store_true')
    parser.add_argument('--optim', type=str)
    parser.add_argument('--throughput', action='store_true')
    parser.add_argument('--pretrained', default=None)

    if PYTORCH_MAJOR_VERSION == 1:
        parser.add_argument('--local_rank', type=int, required=True)

    args, _ = parser.parse_known_args()
    config = get_config(args)
    return args, config


@torch.no_grad()
def evaluate(config, data_loader, model, logger):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    end = time.time()
    for idx, (images, target) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        output = model(images, return_assignments=False)

        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        acc1 = reduce_tensor(acc1)
        acc5 = reduce_tensor(acc5)
        loss = reduce_tensor(loss)

        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))
        acc5_meter.update(acc5.item(), target.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            mem = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'[{idx}/{len(data_loader)}]  '
                f'Time {batch_time.avg:.3f}  '
                f'Loss {loss_meter.avg:.4f}  '
                f'Acc@1 {acc1_meter.avg:.3f}  '
                f'Acc@5 {acc5_meter.avg:.3f}  '
                f'Mem {mem:.0f}MB')

    logger.info(f'==> Acc@1 {acc1_meter.avg:.3f}  Acc@5 {acc5_meter.avg:.3f}')
    return acc1_meter.avg, acc5_meter.avg


if __name__ == '__main__':
    args, config = parse_option()

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
    else:
        rank, world_size = -1, -1

    torch.cuda.set_device(config.LOCAL_RANK)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=config.MODEL.NAME)

    logger.info(config.dump())
    logger.info(json.dumps(vars(args)))

    _, _, _, data_loader_val, _ = build_loader(config)

    logger.info(f"Creating model: {config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)
    model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False)

    logger.info(f"Loading checkpoint from {config.MODEL.RESUME}")
    checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
    state_dict = checkpoint.get('model', checkpoint)
    model.module.load_state_dict(state_dict, strict=False)
    logger.info("Checkpoint loaded.")

    acc1, acc5 = evaluate(config, data_loader_val, model, logger)
    logger.info(f"Final result — Acc@1: {acc1:.3f}  Acc@5: {acc5:.3f}")
