import torch

from src.config.config import Config
from src.models.resnet18 import get_resnet18
from src.models.custom_resnet import CustomResNet


def test_model(model, name):
    print(f"\nTesting {name}...")

    try:
        model = model.to(Config.DEVICE)
        model.eval()

        dummy_input = torch.randn(2, 3, Config.IMAGE_SIZE, Config.IMAGE_SIZE).to(Config.DEVICE)

        with torch.no_grad():
            output = model(dummy_input)

        print(f"{name} working!")
        print(f"Output shape: {output.shape}")

        assert output.shape[1] == Config.NUM_CLASSES, "Output classes mismatch"

    except Exception as e:
        print(f"Error in {name}: {e}")


def main():
    # Test ResNet18
    resnet_model = get_resnet18(Config.NUM_CLASSES)
    test_model(resnet_model, "ResNet18")

    # Test Custom Model
    custom_model = CustomResNet(Config.NUM_CLASSES)
    test_model(custom_model, "Custom ResNet")

if __name__ == "__main__":
    main()


