from roads_fields import predict_folder, MODEL_NAMES

FOLDER = "./dataset/test_images"

for model in ["ResNet50", "MyCNN"]:
    print(f"Predictions with {model} model")
    predict_folder(FOLDER, model)
