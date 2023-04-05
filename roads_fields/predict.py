import glob

from PIL import Image

from roads_fields.models import load_model
from roads_fields.dataloading import get_transforms, HEIGHT, WIDTH, CLASSES


def predict_folder(folder, model_name):
    transforms = get_transforms(train=False)
    model = load_model(model_name)
    for image in sorted(glob.glob(f"{folder}/*.jpeg")):
        im = Image.open(image)
        tensor = transforms(im)
        pred = model(tensor.view(1, 3, HEIGHT, WIDTH))
        print(image, CLASSES[pred.argmax().item()])
