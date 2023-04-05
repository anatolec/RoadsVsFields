from roads_fields import train

print(f"Training MyCNN")
train("MyCNN", epochs=70, lr=0.0001, batch_size=20)

print(f"Training ResNet50")
train("ResNet50", epochs=40, lr=0.0001, batch_size=10)
