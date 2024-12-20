```markdown
# Facial Expression Recognition with FER2013 Dataset

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Data Preparation](#data-preparation)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Model Architecture](#model-architecture)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Visualization and Analysis](#visualization-and-analysis)
- [Saving and Loading the Model](#saving-and-loading-the-model)
- [Confusion Matrix](#confusion-matrix)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

This project implements a **Facial Expression Recognition** system using the FER2013 dataset. The objective is to classify grayscale images of faces into seven distinct emotion categories: 
- `Angry`
- `Disgust`
- `Fear`
- `Happy`
- `Sad`
- `Surprise`
- `Neutral`

The project uses a **Convolutional Neural Network (CNN)** built with TensorFlow and Keras for classification. Additionally, it includes tools for data preprocessing, visualization, and performance analysis.

---

## Dataset

The dataset used in this project is the **FER2013** dataset, which contains:
- **Grayscale images** of size `48x48 pixels`.
- **Emotion labels** corresponding to one of the seven emotion categories listed above.
- **Three subsets** of data:
  - `Training`: Used for training the model.
  - `PublicTest`: Used for validation during training.
  - `PrivateTest`: Used for final model evaluation.

To load the dataset, we use the following code:
```python
data = pd.read_csv(r'C:Downloads\fer2013.csv')
```

The dataset has the following structure:
- The `emotion` column contains the emotion label (integer values from 0 to 6).
- The `pixels` column contains pixel intensity values as a space-separated string.
- The `Usage` column specifies the data split (`Training`, `PublicTest`, or `PrivateTest`).

We analyze the distribution of data across the splits:
```python
data['Usage'].value_counts()
```

---

## Data Preparation

### Converting Data into Usable Format

The dataset provides pixel values as a single string per image, which needs to be converted into a 48x48 numpy array. The following function prepares the data:
```python
def prepare_data(data):
    image_array = np.zeros(shape=(len(data), 48, 48))
    image_label = np.array(list(map(int, data['emotion'])))

    for i, row in enumerate(data.index):
        image = np.fromstring(data.loc[row, 'pixels'], dtype=int, sep=' ')
        image = np.reshape(image, (48, 48))
        image_array[i] = image

    return image_array, image_label
```

This function:
1. Converts the `pixels` column into a numpy array of shape `(48, 48)`.
2. Extracts emotion labels as integers.

We then split the data into training, validation, and test sets:
```python
train_image_array, train_image_label = prepare_data(data[data['Usage'] == 'Training'])
test_image_array, test_image_label = prepare_data(data[data['Usage'] == 'PublicTest'])
val_image_array, val_image_label = prepare_data(data[data['Usage'] == 'PrivateTest'])
```

### Normalizing and Reshaping Data

To prepare the data for the CNN, we normalize pixel values to the range `[0, 1]` and reshape the images to add a channel dimension:
```python
train_images = train_image_array.reshape((train_image_array.shape[0], 48, 48, 1)).astype('float32') / 255
val_images = val_image_array.reshape((val_image_array.shape[0], 48, 48, 1)).astype('float32') / 255
test_images = test_image_array.reshape((test_image_array.shape[0], 48, 48, 1)).astype('float32') / 255
```

The emotion labels are one-hot encoded for categorical classification:
```python
train_labels = to_categorical(train_image_label)
val_labels = to_categorical(val_image_label)
test_labels = to_categorical(test_image_label)
```

---

## Exploratory Data Analysis (EDA)

### Visualizing Examples of Emotions

We define functions to visualize images of specific emotions:

#### Plot Examples of a Specific Emotion
```python
def plot_examples(label=0):
    fig, axs = plt.subplots(1, 5, figsize=(25, 12))
    axs = axs.ravel()
    for i in range(5):
        idx = data[data['emotion'] == label].index[i]
        axs[i].imshow(train_images[idx][:, :, 0], cmap='gray')
        axs[i].set_title(emotions[train_labels[idx].argmax()])
        axs[i].set_xticks([])
        axs[i].set_yticks([])
```

#### Plot One Example for Each Emotion
```python
def plot_all_emotions():
    fig, axs = plt.subplots(1, 7, figsize=(30, 12))
    axs = axs.ravel()
    for i in range(7):
        idx = data[data['emotion'] == i].index[i]
        axs[i].imshow(train_images[idx][:, :, 0], cmap='gray')
        axs[i].set_title(emotions[train_labels[idx].argmax()])
        axs[i].set_xticks([])
        axs[i].set_yticks([])
```

### Compare Emotion Distributions
The following function compares the distributions of emotions in two datasets:
```python
def plot_compare_distributions(array1, array2, title1='', title2=''):
    df_array1 = pd.DataFrame({'emotion': array1.argmax(axis=1)})
    df_array2 = pd.DataFrame({'emotion': array2.argmax(axis=1)})
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].bar(emotions.values(), df_array1['emotion'].value_counts().sort_index(), color='orange')
    axs[0].set_title(title1)
    axs[0].grid()
    
    axs[1].bar(emotions.values(), df_array2['emotion'].value_counts().sort_index())
    axs[1].set_title(title2)
    axs[1].grid()
    
    plt.show()
```

---

## Model Architecture

The CNN for facial expression recognition is built using Keras. The architecture is as follows:
1. **Convolutional Layers**:
   - 3 convolutional layers with ReLU activation.
   - MaxPooling layers for downsampling.
2. **Flatten Layer**: Converts feature maps into a 1D vector.
3. **Fully Connected Layers**:
   - Dense layer with 64 units and ReLU activation.
   - Output layer with 7 units (softmax activation for classification).

The model is defined as:
```python
model = models.Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(MaxPool2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPool2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(7, activation='softmax'))

model.compile(optimizer=Adam(lr=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
```

---

## Model Training

The model is trained using the following function:
```python
def MyModel():
    history = model.fit(
        train_images, train_labels,
        validation_data=(val_images, val_labels),
        class_weight=class_weight,
        epochs=12,
        batch_size=64
    )
```

Class imbalance is handled using class weights:
```python
class_weight = dict(zip(
    range(0, 7),
    (((data[data['Usage'] == 'Training']['emotion'].value_counts()).sort_index()) / len(data[data['Usage'] == 'Training']['emotion'])).tolist()
))
```

---

## Model Evaluation

After training, the model is evaluated on the test set:
```python
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test Accuracy:', test_acc)
```

Predictions are generated using:
```python
pred_test_labels = model.predict(test_images)
```

---

## Visualization and Analysis

### Loss and Accuracy Curves
Training and validation loss/accuracy are plotted:
```python
loss = history.history['loss']
loss_val = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'bo', label='Training Loss')
plt.plot(epochs, loss_val, 'b', label='Validation Loss')
plt.legend()
plt.show()

acc = history.history['accuracy']
acc_val = history.history['val_accuracy']
plt.plot(epochs, acc, 'bo', label='Training Accuracy')
plt.plot(epochs, acc_val, 'b', label='Validation Accuracy')
plt.legend()
plt.show()
```

### Example Predictions
```python
def plot_image_and_emotion(test_image_array, test_image_label, pred_test_labels, image_number):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(test_image_array[image_number], cmap='gray')
    axs[0].set_title(emotions[test_image_label[image_number]])
    axs[1].bar(emotions.values(), pred_test_labels[image_number], color='orange', alpha=0.7)
    plt.show()
```

---

## Confusion Matrix

The confusion matrix visualizes classification performance:
```python
conf_mat = confusion_matrix(test_labels.argmax(axis=1), pred_test_labels.argmax(axis=1))
fig, ax = plot_confusion_matrix(
    conf_mat=conf_mat,
    show_normed=True,
    class_names=emotions.values(),
    figsize=(8, 8)
)
```

---

## Saving and Loading the Model

The trained model is saved for future use:
```python
model.save(r'G:\1-uni\پردازش\saved model')
```

To load the model:
```python
from tensorflow import keras
model = keras.models.load_model(r'G:\1-uni\پردازش\saved model')
```

---

## Future Work

- **Data Augmentation**: Apply transformations to increase dataset size.
- **Advanced Architectures**: Experiment with deeper models or pretrained networks.
- **Real-time Deployment**: Enable real-time emotion recognition with a webcam.

---

## Contributing

Contributions are welcome! Feel free to fork this repository, create a new branch, and submit a pull request.

---

## License

This project is licensed under the MIT License.
```
