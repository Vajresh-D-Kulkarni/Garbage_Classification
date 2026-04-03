import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from src.config.config import Config
from src.data.transforms import get_transforms
from src.models.resnet18 import get_resnet18
from src.models.custom_resnet import CustomResNet
from src.engine.evaluate import evaluate
from src.utils.metrics import plot_confusion_matrix


def get_model():
    if Config.MODEL_NAME == "resnet18":
        return get_resnet18(Config.NUM_CLASSES)
    else:
        return CustomResNet(Config.NUM_CLASSES)


def main():
    device = Config.DEVICE

    # Load model
    model = get_model().to(device)
    model.load_state_dict(torch.load(Config.SAVE_PATH))
    model.eval()

    # Load test dataset
    test_dataset = ImageFolder(
        root=f"{Config.DATA_DIR}/test",
        transform=get_transforms(Config.IMAGE_SIZE)
    )

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Predictions
    preds, labels = evaluate(model, test_loader, device)

    # Plot confusion matrix
    plot_confusion_matrix(
        labels,
        preds,
        test_dataset.classes,
        save_path="outputs/confusion_matrix_test.png"
    )

    for i in range(10):
        print(f"Pred: {test_dataset.classes[preds[i]]} | Actual: {test_dataset.classes[labels[i]]}")


if __name__ == "__main__":
    main()