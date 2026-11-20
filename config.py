"""Global constants."""

# Data
DEFAULT_DATA_DIR = "/home/paul/data/2025_07_07"  # contains 4 cultivars
MIN_TILE_SIZE = 32  # tiles narrower or lower than this (in pixels) are skipped
CLASS_NAMES = ("healthy", "Y-virus")  # label 0, label 1
CLASS_COLORS = ((0, 0, 255), (255, 0, 0))  # RGB box colour per label: blue, red

# Model
IMAGE_SIZE = 192  # every tile is resized to IMAGE_SIZE x IMAGE_SIZE
N_FEATURES = 256  # channels of the backbone output (ResNet-18 layer3)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
