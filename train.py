import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from src.config.config import Config
from src.data.transforms import get_transforms
from src.models.resnet18 import get_resnet18
from src.models.custom_resnet import CustomResNet
from src.engine.train import train_one_epoch
from src.engine.evaluate import evaluate
from src.utils.logger import get_logger
from src.utils.metrics import calculate_metrics, format_metrics
from src.utils.metrics import plot_confusion_matrix

from tqdm import tqdm


def get_model():
    if Config.MODEL_NAME == "resnet18":
        return get_resnet18(Config.NUM_CLASSES)
    else:
        return CustomResNet(Config.NUM_CLASSES)


def prepare_dataloaders():
    train_dir = os.path.join(Config.DATA_DIR, "train")
    val_dir = os.path.join(Config.DATA_DIR, "val")

    transform = get_transforms(Config.IMAGE_SIZE)

    train_dataset = ImageFolder(train_dir, transform=transform)
    val_dataset = ImageFolder(val_dir, transform=transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=2
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=2
    )

    return train_loader, val_loader


def main():
    logger = get_logger()
    logger.info("Training Started")

    device = Config.DEVICE

    # Data
    train_loader, val_loader = prepare_dataloaders()

    # Model
    model = get_model().to(device)

    optimizer = optim.Adam(model.parameters(), lr=Config.LR)
    criterion = nn.CrossEntropyLoss()

    best_f1 = 0

    for epoch in range(Config.EPOCHS):
        logger.info(f"\nEpoch [{epoch+1}/{Config.EPOCHS}]")

        # TRAIN
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        logger.info(f"Train Loss: {train_loss:.4f}")

        # VALIDATION
        preds, labels = evaluate(model, val_loader, device)

        metrics = calculate_metrics(labels, preds)
        metrics_str = format_metrics(metrics)

        logger.info(f"Validation -> {metrics_str}")

        # SAVE BEST MODEL (based on F1)
        if metrics["f1_score"] > best_f1:
            best_f1 = metrics["f1_score"]
            os.makedirs(os.path.dirname(Config.SAVE_PATH), exist_ok=True)
            torch.save(model.state_dict(), Config.SAVE_PATH)
            logger.info("Best model saved (based on F1)!")

    logger.info("Training Completed")

    logger.info("Generating confusion matrix...")

    preds, labels = evaluate(model, val_loader, device)

    class_names = val_loader.dataset.classes

    plot_confusion_matrix(
        labels,
        preds,
        class_names,
        save_path="outputs/confusion_matrix.png"
    )


if __name__ == "__main__":
    main()

