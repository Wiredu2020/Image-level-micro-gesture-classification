from .config import CFG
import torch
from .dataset import label_encodings
from torch import nn
import torch.nn.functional as F
import itertools
import numpy as np
from tqdm.notebook import tqdm
from torch.amp import autocast
from torch.amp import GradScaler
from transformers import DistilBertTokenizer

class AvgMeter:
    """
    Computes and stores the average and current value
    of a given metric.
    """
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        """Resets the meter."""
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        """
        Updates the meter with new value.

        Args:
            val (float): New value to update.
            count (int, optional): Number of samples contributing to the value.
         """
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        """Representation of the metric."""
        text = f"{self.name}: {self.avg:.4f}"
        return text

def get_lr(optimizer):
    """Returns the current learning rate of the optimizer."""
    for param_group in optimizer.param_groups:
        return param_group["lr"]


class KLLoss(nn.Module):
    """
    KL Divergence loss function.
    """
    def __init__(self, error_metric=nn.KLDivLoss(size_average=True, reduce=True)):
        """
        Initializes the KL Divergence loss function.

        Args:
            error_metric (nn.KLDivLoss): KL Divergence loss instance.
        """
        super().__init__()
        self.error_metric = error_metric

    def forward(self, prediction, label):
        """
        Forward pass of the KL Divergence loss function.

        Args:
            prediction (torch.Tensor): Predicted logits.
            label (torch.Tensor): Ground truth labels.

        Returns:
            torch.Tensor: Computed loss value.
        """
        batch_size = prediction.shape[0]
        probs1 = F.log_softmax(prediction, 1)
        probs2 = F.softmax(label * 10, 1)
        loss = self.error_metric(probs1, probs2) * batch_size
        return loss

class CrossEntropyLoss(nn.Module):
    """
    Cross Entropy loss function.
    """
    def __init__(self, reduction='none'):
        """
        Initializes the Cross Entropy loss function.

        Args:
            reduction (str): Method of reducing the loss, 'none', 'mean', or 'sum'.
        """
        super().__init__()
        self.reduction = reduction

    def forward(self, preds, targets):
        """
        Forward pass of the Cross Entropy loss function.

        Args:
            preds (torch.Tensor): Predicted logits.
            targets (torch.Tensor): Ground truth labels.

        Returns:
            torch.Tensor: Computed loss value.
        """
        log_softmax = F.log_softmax(preds, dim=-1)
        loss = (-targets * log_softmax).sum(1)
        if self.reduction == "none":
            return loss
        elif self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        


def train_epoch(model, train_loader, optimizer, lr_scheduler, step, loss_fn):
    """
    Train the model for one epoch.

    Args:
        model (nn.Module): Model instance.
        train_loader (torch.utils.data.DataLoader): Training data loader.
        optimizer (torch.optim.Optimizer): Optimizer instance.
        lr_scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler instance.
        step (str): Whether the scheduler step is per batch or per epoch.
        loss_fn: Loss function instance.

    Returns:
        AvgMeter: Average loss meter for the epoch.
    """
    # Initialize the average loss meter and tqdm progress bar
    loss_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    
    scaler = GradScaler("cuda")
    for batch in tqdm_object:
        # Get batch caption
        caption = batch["caption"]
        # Backpropagation
        optimizer.zero_grad()
        with autocast("cuda"):
            # Use model to encode image and text features
            image_features = model.image_encoder(batch["image"].to(CFG.device))
            text_features = model.text_encoder(input_ids=batch["input_ids"].to(CFG.device), attention_mask=batch["attention_mask"].to(CFG.device))

            # Get embeddings from model projections
            image_embeddings = model.image_projection(image_features)
            text_embeddings = model.text_projection(text_features)

            # Calculating the Loss (Jensen-Shannon (JS) Divergence)
            # Calculate dot product similarity between text embeddings and image embeddings
            texts_similarity = (text_embeddings @ image_embeddings.T) / CFG.temperature
            # Calculate dot product similarity between image embeddings and text embeddings
            images_similarity = (image_embeddings @ text_embeddings.T) / CFG.temperature
            # Generate target matrix
            target = np.array([[1.0 if caption[i] == c else 0.0 for c in caption] for i in range(len(caption))])
            target = torch.tensor(target, dtype=image_embeddings.dtype, device=CFG.device)
            # Calculate JS loss
            loss = (loss_fn(texts_similarity, target) + loss_fn(images_similarity, target))/2


        # Backpropagate and update weights
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        

        # Update loss meter 
        count = len(caption)
        loss_meter.update(loss.item(), count)

        # Update LR scheduler
        if step == "batch":
            lr_scheduler.step(loss_meter.avg)
        # Display progress with average training loss and learning rate
        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))

    return loss_meter


def valid_epoch(model, valid_loader, loss_fn):
    """
    Validate the model.

    Args:
        model (nn.Module): Model instance.
        valid_loader (torch.utils.data.DataLoader): Validation data loader
        loss_fn: Loss function instance.

    Returns:
        AvgMeter: Average loss meter for the validation.
    """

    # Initialize the average loss meter and tqdm progress bar
    Val_loss_meter = AvgMeter()
    val_acc, val_acc5 = 0.0, 0.0
    
    tqdm_object = tqdm(valid_loader, total=len(valid_loader))
    total_number = len(tqdm_object)* CFG.batch_size
    tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
    encoded_captions = tokenizer(list(label_encodings.values()), padding=True, truncation=True, max_length=CFG.max_length)
    item = {key: torch.tensor(values) for key, values in encoded_captions.items()}

    # Disable gradients
    for batch in tqdm_object:
        caption = batch["caption"]
        with torch.no_grad():
            # Get batch caption
        
            # Use model to encode image and text features
            image_features = model.image_encoder(batch["image"].to(CFG.device))
            text_features = model.text_encoder(input_ids=batch["input_ids"].to(CFG.device), attention_mask=batch["attention_mask"].to(CFG.device))
            test_text_features = model.text_encoder(input_ids=item["input_ids"].to(CFG.device), attention_mask=item["attention_mask"].to(CFG.device))
            # Get embeddings from model projections
            image_embeddings = model.image_projection(image_features)
            text_embeddings = model.text_projection(text_features)
            test_text_embeddings = model.text_projection(test_text_features)
            # Calculating the Loss (Jensen-Shannon (JS) Divergence)
            # Calculate dot product similarity between text embeddings and image embeddings
            texts_similarity = (text_embeddings @ image_embeddings.T) / CFG.temperature
            # Calculate dot product similarity between image embeddings and text embeddings
            images_similarity = (image_embeddings @ text_embeddings.T) / CFG.temperature
            # Generate target matrix
            target = np.array([[1.0 if caption[i] == c else 0.0 for c in caption] for i in range(len(caption))])

            target = torch.tensor(target, dtype=image_embeddings.dtype, device=CFG.device)
            
            # Calculate JS loss
            loss = (loss_fn(texts_similarity, target) + loss_fn(images_similarity, target))/2
            
            dot_similarity = image_embeddings @ test_text_embeddings.T
            
            values, indices_pred = torch.topk(dot_similarity.squeeze(0), 5)
            indices_pred = indices_pred.detach().cpu().numpy()
            for idx, tar in enumerate(caption):
                pred_indices = [label_encodings[a + 1] for a in indices_pred[idx]]
                if label_encodings[indices_pred[idx][0] + 1] == tar:
                    val_acc +=1
                    val_acc5 += 1
                elif tar in pred_indices:
                    val_acc5 += 1

        # Update loss meter
        
        count = len(caption)
        Val_loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(valid_loss=Val_loss_meter.avg)
                        # Calculate the dot product similarity between image and text embeddings
        

    # Print the top-1 and top-5 accuracies
    print('Val_Acc@1 == {:.2f}%'.format((val_acc/total_number) * 100))
    print('Val_Acc@5 == {:.2f}%'.format((val_acc5/total_number) * 100))

    # Return loss meter and top-1 accuracy
    return Val_loss_meter, val_acc/total_number ,val_acc5/total_number