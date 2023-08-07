import torch
import torch.distributed
import torch.nn as nn
import torch.optim as optim

import os

from typing import Any
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel

# my version
def save_checkpoints(
    save_checkpoints_dir, check_point_params, checkpoint_prefix=None
):
    # type: (str, dict, int, nn.Module, optim.Optimizer, optim.lr_scheduler.LambdaLR, str) -> None
    """
    Save checkpoints. Execute at the end of a iteration.

    Arguments:
        save_checkpoints_dir (str):
        cur_epoch (int):
        cur_step (int):optim.lr_scheduler.LambdaLR, str
        model (nn.Module):
        optimizer (optim.Optimizer):
        scheduler (optim.lr_scheduler.LambdaLR):
        checkpoint_prefix (str, optional):
    """

    checkpoint = {
        "epoch": check_point_params['epochs'],
        "model_state_dict": check_point_params['state_dict'].state_dict(),
        'model_optimizer': check_point_params['model_optimizer'].state_dict()
    }

    checkpoint_file = (f"{checkpoint_prefix}_" if checkpoint_prefix else "") + "check_point.ckpt"
    save_path = os.path.join(save_checkpoints_dir, checkpoint_file)
    torch.save(checkpoint, save_path, _use_new_zipfile_serialization=False)


# my version
def load_checkpoints(load_checkpoints_path, train_epoch, model, bert_optimizer, model_optimizer, checkpoint_prefix=None):
    # type: (str, nn.Module, optim.Optimizer, optim.lr_scheduler.LambdaLR) -> (int, int, nn.Module, optim.Optimizer, optim.lr_scheduler.LambdaLR)
    """
    Load checkpoints. Execute at the start of iteration.

    Arguments:
        load_checkpoints_path (str):
        model (nn.Module):
        optimizer (optim.Optimizer):
        scheduler (optim.lr_scheduler.LambdaLR):
    """
    checkpoint_file = (f"{checkpoint_prefix}_" if checkpoint_prefix else "") + f"epoch_{train_epoch}.ckpt"
    checkpoint = torch.load(os.path.join(load_checkpoints_path, checkpoint_file), map_location="cpu")

    epoch = checkpoint["epoch"]

    model.load_state_dict(checkpoint["model_state_dict"])
    bert_optimizer = checkpoint["bert_optimizer"],
    model_optimizer = checkpoint['model_optimizer'],

    return model, bert_optimizer, model_optimizer,loss, epoch


def load_checkpoint_attribute_from_key(load_checkpoints_path, any_obj, key):
    # type: (str, Any, str) -> Any

    checkpoint = torch.load(load_checkpoints_path, map_location="cpu")

    value = checkpoint[key]
    if isinstance(value, int):
        any_obj = value
    else:
        any_obj.load_state_dict(value)

    return any_obj


def save_model_state_dict(save_state_dict_dir, state_dict_file, model):
    # type: (str, str, (nn.Module, DistributedDataParallel)) -> None
    """
    Save model state dict. Note this function is not for reproduce or continuing train.

    Arguments:
        model (nn.Module):
        save_state_dict_dir (str):
        state_dict_file (str):
    """

    model_to_save = model.module if hasattr(model, 'module') else model
    save_path = os.path.join(save_state_dict_dir, state_dict_file)
    torch.save(model_to_save.state_dict(), save_path, _use_new_zipfile_serialization=False)


def load_best_model(save_model_path, best_model_name, model):
    device = torch.device('cuda')

    model.load_state_dict(torch.load(os.path.join(save_model_path, best_model_name), map_location="cpu"))

    model.to(device)

    return model

def write_log(save_log_dir, log_file, iter_bar):
    # type: (str, str, tqdm) -> None
    log_save_path = os.path.join(save_log_dir, log_file)
    with open(log_save_path, "a+") as fp:
        fp.write(f"{iter_bar.postfix}\n")
