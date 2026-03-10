from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.utils.tensorboard as tb
from utils import FaceDiseaseDataset, compute_accuracy, load_data
from models import FaceDiseaseCNN, ClassficationLoss

def train(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader,
          criterion: torch.nn.Module, device: torch.device,
          optimizer: torch.optim.Optimizer,
          global_step: int,
          logger: tb.SummaryWriter) -> tuple[list[float], int]:

    model.train()
    train_accuracy = []
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        train_accuracy.append(compute_accuracy(outputs, labels).item())

        loss = criterion(outputs, labels)
        logger.add_scalar("Loss/Train", loss.item(), global_step)
        loss.backward()
        optimizer.step()
        global_step += 1
    return train_accuracy, global_step

def evaluate(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader,
             criterion: torch.nn.Module, device: torch.device,
             global_step: int,
             logger: tb.SummaryWriter) -> list[float]:

    with torch.inference_mode():
        model.eval()
        val_accuracy = []
        val_losses = []
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            val_accuracy.append(compute_accuracy(outputs, labels).item())
            val_losses.append(criterion(outputs, labels).item())

    avg_val_loss = sum(val_losses) / len(val_losses)
    logger.add_scalar("Loss/Validation", avg_val_loss, global_step)
    return val_accuracy


