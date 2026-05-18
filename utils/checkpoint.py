import os
import torch


def load_checkpoint(config, model, optimizer, lr_scheduler, loss_scaler, logger):
    logger.info(f"==============> Resuming from {config.MODEL.RESUME}")
    if config.MODEL.RESUME.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            config.MODEL.RESUME, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    logger.info(msg)
    max_accuracy = 0.0
    if not config.EVAL_MODE and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        config.defrost()
        config.TRAIN.START_EPOCH = checkpoint['epoch'] + 1
        config.freeze()
        if 'scaler' in checkpoint:
            loss_scaler.load_state_dict(checkpoint['scaler'])
        logger.info(f"=> loaded successfully '{config.MODEL.RESUME}' (epoch {checkpoint['epoch']})")
        if 'max_accuracy' in checkpoint:
            max_accuracy = checkpoint['max_accuracy']
    del checkpoint
    torch.cuda.empty_cache()
    return max_accuracy


def load_pretrained(config, model, logger):
    logger.info(f"==============> Loading weight {config.MODEL.PRETRAINED} for fine-tuning")
    checkpoint = torch.load(config.MODEL.PRETRAINED, map_location='cpu')
    state_dict = checkpoint['model']

    for k in [k for k in state_dict if "relative_position_index" in k]:
        del state_dict[k]
    for k in [k for k in state_dict if "relative_coords_table" in k]:
        del state_dict[k]
    for k in [k for k in state_dict if "attn_mask" in k]:
        del state_dict[k]

    for k in [k for k in state_dict if "relative_position_bias_table" in k]:
        pretrained = state_dict[k]
        current = model.state_dict()[k]
        L1, nH1 = pretrained.size()
        L2, nH2 = current.size()
        if nH1 != nH2:
            logger.warning(f"Head mismatch in {k}, skipping")
        elif L1 != L2:
            S1, S2 = int(L1 ** 0.5), int(L2 ** 0.5)
            resized = torch.nn.functional.interpolate(
                pretrained.permute(1, 0).view(1, nH1, S1, S1), size=(S2, S2), mode='bicubic')
            state_dict[k] = resized.view(nH2, L2).permute(1, 0)

    for k in [k for k in state_dict if "absolute_pos_embed" in k]:
        pretrained = state_dict[k]
        current = model.state_dict()[k]
        _, L1, C1 = pretrained.size()
        _, L2, C2 = current.size()
        if C1 == C2 and L1 != L2:
            S1, S2 = int(L1 ** 0.5), int(L2 ** 0.5)
            pretrained = pretrained.reshape(-1, S1, S1, C1).permute(0, 3, 1, 2)
            resized = torch.nn.functional.interpolate(pretrained, size=(S2, S2), mode='bicubic')
            state_dict[k] = resized.permute(0, 2, 3, 1).flatten(1, 2)

    Nc1 = state_dict['head.bias'].shape[0]
    Nc2 = model.head.bias.shape[0]
    if Nc1 != Nc2:
        torch.nn.init.constant_(model.head.bias, 0.)
        torch.nn.init.constant_(model.head.weight, 0.)
        del state_dict['head.weight']
        del state_dict['head.bias']
        logger.warning("Classifier head mismatch — re-initializing head to zero")

    msg = model.load_state_dict(state_dict, strict=False)
    logger.warning(msg)
    logger.info(f"=> loaded successfully '{config.MODEL.PRETRAINED}'")
    del checkpoint
    torch.cuda.empty_cache()


def save_checkpoint(config, epoch, model, max_accuracy, optimizer, lr_scheduler, loss_scaler, logger):
    save_path = os.path.join(config.OUTPUT, f'ckpt_epoch_{epoch}.pth')
    logger.info(f"{save_path} saving...")
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'max_accuracy': max_accuracy,
        'scaler': loss_scaler.state_dict(),
        'epoch': epoch,
        'config': config,
    }, save_path)
    logger.info(f"{save_path} saved.")


def auto_resume_helper(output_dir):
    checkpoints = [ckpt for ckpt in os.listdir(output_dir) if ckpt.endswith('pth')]
    print(f"All checkpoints in {output_dir}: {checkpoints}")
    if checkpoints:
        latest = max([os.path.join(output_dir, d) for d in checkpoints], key=os.path.getmtime)
        print(f"Latest checkpoint: {latest}")
        return latest
    return None
