# Native Segmentation Vision Transformer

Unofficial implementation of the paper [Native Segmentation Vision Transformer](https://arxiv.org/pdf/2505.16993).

This repository was built because I could not find an official open-source implementation when reproducing the paper. The project started from a cloned Swin Transformer codebase, but the implementation here was heavily modified to reproduce the Native Segmentation Vision Transformer idea in this repository.

`main.py` is the main training and evaluation entry point.

## Notes

- This is not the official code release from the paper.
- The codebase was originally bootstrapped from Swin Transformer and then substantially changed.
- Requires PyTorch 2.x. Use `torchrun` to launch training (see commands below).
- The commands below use repository-relative config paths. Replace checkpoint and dataset paths with the ones from your environment.

## Result

(177/300 epochs)

- ImageNet-1K validation Top-1: `76.150%`
- ImageNet-1K validation Top-5: `93.445%`
- Best recorded accuracy in log: `76.27%`

These numbers are from `output/Senatra/default/log_rank2.txt`.

## Pretrained Checkpoint

A trained checkpoint can be downloaded from Google Drive:

[Download `.pth` checkpoint](https://drive.google.com/file/d/1kH9YIwW8MOjCgZIUofOdRvsuAGoed3BQ/view?usp=drive_link)

After downloading the `.pth` file, you can plug its path into `--resume` and run evaluation immediately.

## Installation

```bash
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu121
```

## Training And Evaluation

### Single-GPU training

```bash
CUDA_VISIBLE_DEVICES=0 torchrun \
  --nproc_per_node=1 \
  --master-port 1215 \
  main.py \
  --cfg configs/swin/Senatra.yaml \
  --data-path /path/to/imagenet \
  --batch-size 64
```

### Single-GPU evaluation

```bash
CUDA_VISIBLE_DEVICES=0 torchrun \
  --nproc_per_node=1 \
  --master-port 1215 \
  main.py \
  --eval \
  --cfg configs/swin/Senatra.yaml \
  --resume /path/to/checkpoint.pth \
  --data-path /path/to/imagenet
```

### Multi-GPU training (e.g. 4 GPUs)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
  --nproc_per_node=4 \
  --master-port 1215 \
  main.py \
  --cfg configs/swin/Senatra.yaml \
  --data-path /path/to/imagenet \
  --batch-size 64
```

### Multi-GPU evaluation (e.g. 4 GPUs)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
  --nproc_per_node=4 \
  --master-port 1215 \
  main.py \
  --eval \
  --cfg configs/swin/Senatra.yaml \
  --resume /path/to/checkpoint.pth \
  --data-path /path/to/imagenet
```

## Key Arguments

| Argument | Description |
|---|---|
| `--cfg` | Path to config file |
| `--data-path` | Path to ImageNet dataset root |
| `--batch-size` | Per-GPU batch size |
| `--resume` | Path to checkpoint `.pth` for resuming or evaluation |
| `--eval` | Run evaluation only (no training) |
| `--output` | Output directory root (default: `output`) |
| `--tag` | Experiment tag (default: `default`) |

## Acknowledgement

This repository is based on a cloned Swin Transformer codebase and was heavily modified for this unofficial Native Segmentation Vision Transformer implementation.
