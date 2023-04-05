# Roads vs Fields model

This repo contains the code to train a computer vision model that
predicts whether an image contains a Road or a Field.

It uses PyTorch to build and train Convolutional Neural Networks.

Sample images are contained in the `dataset` directory.


## Usage
1. To install dependencies run `pip install -r requirements.txt`
2. To train the model run ``python train.py``, it will train both a pretrained model and a custom model on the above training dataset
3. To run predictions, run ``python predict.py``. You may edit predict.py to change the directory to be predicted.


## Write-up

This exercice was comprised of 4 steps:
1. Exploration
2. Preprocessing
3. Modelisation
4. Results


### Exploration

First, we did a quick exploration in a jupyter notebook (see `Image exploration.ipynb`).
Thanks to that, we identified 3 interesting points:
1. Most of our images are in landscape format, with an aspect ratio around 1.5. This was actually expected given the task.
We will preserve this format in our modelisation as it seems to be inherent to this kind of data.
2. There are much more roads samples than fields samples. It can be due to 2 facts:
   1. The "real world" contains more roads than fields, in other words it is expected that our model will encounter more roads than fields
   2. The "real world" actually contains the same proportion of roads and fields, but the sample dataset is imbalanced compared to the real world
   
   --> There is no reason to think that the real world contains more roads than fields, so we will assume the second possibility and we will need to cope with our unbalanced dataset.
   We can do so by 1 - manually rebalancing the datasets when loading them or 2 - using the dataset as it is, but using the class weights when measuring model accuracy.
3. Two field images were wrongly classified as roads, so we moved them to the roads dataset (`3_mis.jpg` and `5_mis.jpg`)


### Preprocessing

Then, we wrote the code to load and preprocess our dataset (see ``roads_fields/dataloading.py``)

A few notable points here:
1. When loading the training dataset, we apply random augmentations. That is, we randomly flip the images, rotate them, apply image filters etc.
This step prevents our model from overfitting on our train data, and it allows it to truly focus on what makes a road, a road and a field, a field.
2. We then resize the images to the size we chose at the exploration step above
3. We normalize the resulting tensors so that their values fall between 0 and 1. We do so by using the recommended ImageNet normalization parameters.
4. Finally, we create our data loaders by using 25% of the data to validate our model. We also compute here the class weights, to be passed to our loss criterion


### Modelisation

We defined our models in ``roads_fields/models.py``. We chose to build a custom Convolutional network. And to assess its performance, we also used a state-of-the-art pretrained model.
1. Our custom CNN is composed of 3 convolutional layers followed by 2 fully connected layers. The number of neurons in each layer was chosen by trial and error (a more systematic approach could include a Grid Search on the number of neurons and layers).
Each convolution layer uses a 3x3 kernel size as it is the most popular chose and is followed by a max-pooling layer to reduce the dimension and accelerate the model training.
We chose ReLu as our activation function for all layers. It is unanimously recognized as the go-to solution for CNNs.
2. ResNet50 has been one of the leading models in image classification. After loading it, we freeze the pretrained weights and redefine the fully connected layer to fit our usecase.


Both our models were trained with the same criterion and optimizer:
- __Criterion__ : For classification purposes, we don't have much choice. The vastly preferred way is to use CrossEntropyLoss, however, we pass it the class weights because our dataset in unbalanced.
- __Optimizer__ : We used the well-known Adam optimizer as it usually converges faster than SGD. It is recommended in most papers for image classification (for example : https://opt-ml.org/papers/2021/paper53.pdf). 
We initialized the learning rate at 0.001 (1e3), but we had to decrease it to 0.0001 (1e4) as the losses were fluctuating too much.


### Results

Our CNN model converged after 70 epochs, reaching an accuracy around ~92%
![MyCNN Loss](./roads_fields/output/MyCNN/losses.png)
![MyCNN Accuracy](./roads_fields/output/MyCNN/accuracies.png)

In comparison, ResNet50 converged after 40 epochs, reaching an accuracy around ~95%
![MyCNN Loss](./roads_fields/output/ResNet50/losses.png)
![MyCNN Accuracy](./roads_fields/output/ResNet50/accuracies.png)

We then applied our MyCNN model to the 10 test_images, it reached a 100% accuracy on this set :
1. fields
2. roads
3. roads
4. fields
5. roads
6. roads
7. roads
8. roads
9. fields
10. fields