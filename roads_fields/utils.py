from pathlib import Path

ROOT = Path(__file__).parent.parent
DATASET = ROOT / "dataset"
TRAIN_SET = DATASET / "train"
TEST_SET = DATASET / "test_images"
PACKAGE_ROOT = ROOT / "roads_fields"
OUTPUT = PACKAGE_ROOT / "output"
