# Native Segmentation Vision Transformer

Unofficial implementation of the paper [Native Segmentation Vision Transformer](https://arxiv.org/pdf/2505.16993).

This repository was built because I could not find an official open-source implementation when reproducing the paper. The project started from a cloned Swin Transformer codebase, but the implementation here was heavily modified to reproduce the Native Segmentation Vision Transformer idea in this repository.

## Notes

- This is not the official code release from the paper.
- The codebase was originally bootstrapped from Swin Transformer and then substantially changed.
- Requires PyTorch 2.x. Use `torchrun` to launch all scripts.
- `--batch-size` is per-GPU batch size.
- `--master-port` must be a free port on your machine.

## Result

(177/300 epochs)

- ImageNet-1K validation Top-1: `76.150%`
- ImageNet-1K validation Top-5: `93.445%`
- Best recorded accuracy in log: `76.27%`

## Pretrained Checkpoint

[Download `nativeseg.pth`](https://drive.google.com/file/d/1kH9YIwW8MOjCgZIUofOdRvsuAGoed3BQ/view?usp=drive_link)

## Installation

```bash
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu121
```

## Project Structure

```
train.py                      # Training (and resume/eval via --eval)
inference_for_classification.py  # Classification evaluation on ImageNet val
segmentation_test.py          # Segmentation visualization (markov / class_projection)
models/
  swin_transformer.py         # Senatra and Senatra_segmentation model definitions
utils/
  checkpoint.py               # load/save checkpoint helpers
  training.py                 # NativeScaler, reduce_tensor, grad norm helpers
configs/swin/
  Senatra.yaml                # Config for training / markov visualization
  Senatra_segmentation.yaml   # Config for class_projection visualization
```

---

## Training

### Single GPU

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master-port 1215 train.py \
  --cfg configs/swin/Senatra.yaml \
  --data-path /path/to/imagenet \
  --batch-size 64
```

### Multi GPU (e.g. 4 GPUs)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master-port 1215 train.py \
  --cfg configs/swin/Senatra.yaml \
  --data-path /path/to/imagenet \
  --batch-size 64
```

### Resume from checkpoint

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master-port 1215 train.py \
  --cfg configs/swin/Senatra.yaml \
  --resume output/Senatra/default/ckpt_epoch_X.pth \
  --data-path /path/to/imagenet \
  --batch-size 64
```

### Evaluation only

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master-port 1215 train.py \
  --eval \
  --cfg configs/swin/Senatra.yaml \
  --resume nativeseg.pth \
  --data-path /path/to/imagenet
```

---

## Classification Inference

Runs evaluation on the full ImageNet val set and reports Top-1 / Top-5 accuracy.

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master-port 1215 inference_for_classification.py \
  --cfg configs/swin/Senatra.yaml \
  --resume nativeseg.pth \
  --data-path /path/to/imagenet \
  --batch-size 64
```

---

## Segmentation Visualization

Two visualization modes are available via `--mode`.

### Mode 1: Markov chain assignment map

Visualizes which segment token each patch is assigned to through the Markov upsampling chain.
Uses the standard `Senatra` model.

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master-port 1216 segmentation_test.py \
  --mode markov \
  --cfg configs/swin/Senatra.yaml \
  --resume nativeseg.pth \
  --data-path /path/to/imagenet \
  --batch-size 8 \
  --max-batches 5
```

Results saved to `./segmentation/assignment_chain/`.

### Mode 2: Class projection segmentation map

Projects segment token class predictions back to the patch grid to produce a semantic segmentation map.
Uses the `Senatra_segmentation` model.

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master-port 1216 segmentation_test.py \
  --mode class_projection \
  --cfg configs/swin/Senatra_segmentation.yaml \
  --resume nativeseg.pth \
  --data-path /path/to/imagenet \
  --batch-size 8 \
  --max-batches 5
```

Results saved to `./segmentation/class_projection/`.

---

## Key Arguments

| Argument | Description |
|---|---|
| `--cfg` | Path to config YAML |
| `--data-path` | Path to ImageNet dataset root |
| `--resume` | Path to checkpoint `.pth` |
| `--batch-size` | Per-GPU batch size |
| `--eval` | (train.py only) Run evaluation only, no training |
| `--max-batches` | (segmentation_test.py only) Number of batches to visualize |
| `--mode` | (segmentation_test.py only) `markov` or `class_projection` |
| `--output` | Output directory root (default: `output`) |
| `--tag` | Experiment tag (default: `default`) |

## Acknowledgement

This repository is based on a cloned Swin Transformer codebase and was heavily modified for this unofficial Native Segmentation Vision Transformer implementation.
