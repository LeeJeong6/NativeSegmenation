# Markov chain assignment map (uses Senatra config):
# CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master-port 1216 segmentation_test.py \
#   --mode markov --cfg configs/swin/Senatra.yaml --resume nativeseg.pth \
#   --data-path /raid/Datasets/imagenet/ --batch-size 8 --max-batches 5
#
# Class projection segmentation map (uses Senatra_segmentation config):
# CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master-port 1216 segmentation_test.py \
#   --mode class_projection --cfg configs/swin/Senatra_segmentation.yaml --resume nativeseg.pth \
#   --data-path /raid/Datasets/imagenet/ --batch-size 8 --max-batches 5

import os
import json
import random
import argparse
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F
import matplotlib.pyplot as plt

from config import get_config
from models import build_model
from data import build_loader
from logger import create_logger

PYTORCH_MAJOR_VERSION = int(torch.__version__.split('.')[0])

SAVE_ROOT = "./segmentation"
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def denormalize_imagenet(images):
    mean = IMAGENET_MEAN.to(device=images.device, dtype=images.dtype)
    std = IMAGENET_STD.to(device=images.device, dtype=images.dtype)
    return (images * std + mean).clamp(0, 1)


def compose_assignment_chain(a_ups_list):
    chain = a_ups_list[0]
    for mat in a_ups_list[1:]:
        chain = chain @ mat
    return chain


def save_figures(images, vis_map, batch_idx, rank, save_dir, title):
    os.makedirs(save_dir, exist_ok=True)
    images_denorm = denormalize_imagenet(images).cpu()
    vis_map = vis_map.cpu()

    for i in range(images_denorm.shape[0]):
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(images_denorm[i].permute(1, 2, 0).numpy())
        ax[0].set_title("Original")
        ax[0].axis("off")
        ax[1].imshow(vis_map[i].numpy())
        ax[1].set_title(title)
        ax[1].axis("off")
        path = os.path.join(save_dir, f"rank{rank:02d}_batch{batch_idx:05d}_sample{i:02d}.png")
        plt.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")


@torch.no_grad()
def run_markov(config, data_loader, model, max_batches, rank):
    """Visualize Markov chain assignment map (segment token assignment per patch)."""
    save_dir = os.path.join(SAVE_ROOT, "assignment_chain")
    model.eval()

    for idx, (images, _) in enumerate(data_loader):
        if idx >= max_batches:
            break
        images = images.cuda(non_blocking=True)

        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            result = model(images, return_assignments=True)
        # Senatra returns (logits, a_ups_list, a_down_list) when return_assignments=True
        _, a_ups_list, _ = result

        a_total = compose_assignment_chain(a_ups_list)    # (B, N_patch, N_segment)
        bsz, n_patch, _ = a_total.shape
        grid = int(n_patch ** 0.5)

        # argmax over segment axis → which segment each patch belongs to
        seg_map = a_total.argmax(dim=-1).reshape(bsz, grid, grid).float().unsqueeze(1)
        seg_map = F.interpolate(seg_map, size=tuple(images.shape[-2:]), mode="nearest")[:, 0]

        save_figures(images, seg_map, idx, rank, save_dir, title="Markov Assignment")
        print(f"[{idx+1}/{max_batches}] markov batch done")


@torch.no_grad()
def run_class_projection(config, data_loader, model, max_batches, rank):
    """Visualize class projection segmentation map (per-patch class prediction)."""
    save_dir = os.path.join(SAVE_ROOT, "class_projection")
    model.eval()

    for idx, (images, _) in enumerate(data_loader):
        if idx >= max_batches:
            break
        images = images.cuda(non_blocking=True)

        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            # Senatra_segmentation returns (tokens, logits, a_ups_list, a_down_list)
            tokens, _, a_ups_list, _ = model(images)

        a_total = compose_assignment_chain(a_ups_list)    # (B, N_patch, N_segment)
        bsz, n_segment, dim = tokens.shape
        num_classes = model.head.out_features

        tokens_flat = tokens.reshape(bsz * n_segment, dim)
        seg_logits = model.head(tokens_flat).reshape(bsz, n_segment, num_classes)
        patch_logits = a_total @ seg_logits               # (B, N_patch, C)

        grid = int(patch_logits.shape[1] ** 0.5)
        patch_logits = patch_logits.permute(0, 2, 1).reshape(bsz, num_classes, grid, grid)
        seg_map = F.interpolate(patch_logits, size=tuple(images.shape[-2:]), mode="bilinear").argmax(dim=1).float()

        save_figures(images, seg_map, idx, rank, save_dir, title="Class Projection")
        print(f"[{idx+1}/{max_batches}] class_projection batch done")


def parse_option():
    parser = argparse.ArgumentParser('Senatra segmentation visualization', add_help=False)
    parser.add_argument('--mode', type=str, required=True, choices=['markov', 'class_projection'],
                        help='markov: Markov chain assignment map  |  class_projection: per-patch class map')
    parser.add_argument('--cfg', type=str, required=True, metavar='FILE', help='path to config file')
    parser.add_argument("--opts", default=None, nargs='+')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--data-path', type=str, default='/raid/Datasets/imagenet/')
    parser.add_argument('--resume', type=str, default='nativeseg.pth', help='checkpoint path')
    parser.add_argument('--max-batches', type=int, default=5, help='number of batches to visualize')
    parser.add_argument('--zip', action='store_true')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'])
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

    logger.info(f"Loading checkpoint from {config.MODEL.RESUME}")
    checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
    state_dict = checkpoint.get('model', checkpoint)
    model.load_state_dict(state_dict, strict=False)
    logger.info("Checkpoint loaded.")

    rank_id = dist.get_rank()
    max_batches = args.max_batches

    if args.mode == 'markov':
        logger.info(f"Mode: Markov chain assignment map — {max_batches} batches")
        run_markov(config, data_loader_val, model, max_batches, rank_id)
        logger.info(f"Done. Results saved to {SAVE_ROOT}/assignment_chain/")
    else:
        logger.info(f"Mode: Class projection segmentation map — {max_batches} batches")
        run_class_projection(config, data_loader_val, model, max_batches, rank_id)
        logger.info(f"Done. Results saved to {SAVE_ROOT}/class_projection/")
