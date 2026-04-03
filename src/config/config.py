import os

class Config:
    DATA_DIR = 'data/garbage_classification_processed'
    NUM_CLASSES = 12
    IMAGE_SIZE = 224
    BATCH_SIZE = 32
    EPOCHS = 20
    LR = 1e-3
    DEVICE = 'cuda' if os.environ.get('USE_CUDA', '0') == '1' else 'cpu'

    MODEL_NAME = 'resnet18'
    SAVE_PATH = 'outputs/models/model.pth'
