import torch
from PIL import Image
from torchvision.datasets import ImageFolder

from src.config.config import Config
from src.data.transforms import get_transforms
from src.models.resnet18 import get_resnet18
from src.models.custom_resnet import CustomResNet


def get_model():
    if Config.MODEL_NAME == "resnet18":
        return get_resnet18(Config.NUM_CLASSES)
    else:
        return CustomResNet(Config.NUM_CLASSES)


def predict_image(image_path):
    device = Config.DEVICE

    model = get_model().to(device)
    model.load_state_dict(torch.load(Config.SAVE_PATH))
    model.eval()

    transform = get_transforms(Config.IMAGE_SIZE)

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        _, pred = torch.max(output, 1)

    return pred.item()


if __name__ == "__main__":
    img_path = "data/garbage_classification_processed/test/battery/battery157.jpg"
    pred_class = predict_image(img_path)
    class_names = ImageFolder(f"{Config.DATA_DIR}/train").classes
    print(f"Prediction: {class_names[pred_class]}")
    # print(f"Predicted Class Index: {pred_class}")