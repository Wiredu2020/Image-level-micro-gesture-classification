import os
import torch


def save_ckp(state, epoch, checkpoint_dir):
    """
    Save model checkpoint.

    Args:
        state (dict): Model state dictionary.
        epoch (int): Current epoch.
        checkpoint_dir (str): Directory to save checkpoints.
    """
    save_path = os.path.join(checkpoint_dir, f'ckpt_epoch_{epoch}.pth')
    torch.save(state, save_path)

def load_ckp(model,loc_cpk = "utils/kl2captioncheckpoints/ckpt_epoch_1.pth",trian_opt = False):
    # 3. Load states
    checkpoint = torch.load(loc_cpk, map_location='cpu')  # or cpu

    model.load_state_dict(checkpoint['model'])
    # optimizer.load_state_dict(checkpoint['optimizer'])
    # lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    # epoch = checkpoint['epoch']
    
    # if train_opt:
    #     return model, m
    return model