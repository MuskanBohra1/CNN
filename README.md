# Image Classification Using Convolutional Neural Networks (CNN)

This project demonstrates how to build a **Convolutional Neural Network (CNN)** to perform **binary image classification**. The goal is to train a model to recognize and classify images into two distinct categories. The model uses deep learning techniques in Python, leveraging **TensorFlow** and **Keras**.

This project was completed as part of the **Big Data Analytics PGDM** program at **FORE School of Management**.

---

##  Objective

The objective of this project is to create a CNN model that can classify images into two categories (e.g., Class A vs. Class B). The primary steps involved include:

1. **Data Preprocessing**: Cleaning and preparing image data.
2. **CNN Model Construction**: Building a neural network with convolutional layers to extract spatial features.
3. **Training and Evaluation**: Training the model and evaluating it on unseen test data.
4. **Predictions**: Using the trained model to classify new, unseen images.

---

## Dataset

The dataset used for this project is composed of images organized into two classes. It includes:

- **Training data**: A set of labeled images used to train the model.
- **Test data**: A set of images the model hasn't seen before, used for evaluation.

The images are preprocessed and resized to ensure consistency for the CNN model. For data augmentation, the dataset undergoes transformations such as random rotations, flips, and zooms to enhance model generalization.

---

## Model Architecture

The CNN model used in this project follows a standard deep learning architecture for image classification:

- **Input Layer**: Images of shape `(150, 150, 3)` (height, width, color channels)
- **Convolutional Layers**: Extract features from the image using 32, 64, and 128 filters.
- **Max Pooling**: Reduces dimensionality and retains important features.
- **Dropout Layers**: Prevents overfitting by randomly deactivating neurons during training.
- **Fully Connected Layers**: Flattens the features into a vector for final classification.
- **Output Layer**: A single neuron with a sigmoid activation for binary classification (Class A or Class B).

---

## Results

The model was evaluated on test data, achieving a **test accuracy of ~92%**. The evaluation includes several metrics:

- **Accuracy**: Percentage of correct classifications.
- **Loss**: Measure of error between predicted and actual labels.
- **Confusion Matrix**: Highlights true positives, false positives, true negatives, and false negatives.

---

## Future Enhancements

1. **Multi-Class Classification**  
   Extend the model to handle more than two categories, such as distinguishing between multiple object types.

2. **Data Augmentation**  
   Use more sophisticated augmentation techniques like **CutMix**, **MixUp**, or **Random Erasing** to further improve model robustness.

3. **Transfer Learning**  
   Experiment with pretrained models (e.g., **VGG16**, **ResNet50**) to leverage pre-learned features and fine-tune the model for better performance.

4. **Web App Deployment**  
   Deploy the trained model as a web application using **Flask** or **Streamlit**, allowing users to upload images and get predictions in real-time.
