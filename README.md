```markdown
# Emotion Recognition from Facial Expressions

This project implements a Convolutional Neural Network (CNN) to recognize emotions from facial expressions in images. It utilizes the FER2013 dataset, which contains images of faces labeled with seven different emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral.

## Installation

To run this code, you need to have Python installed along with the following libraries. You can install them using pip:

```bash
pip install pandas numpy matplotlib scikit-learn mlxtend tensorflow keras
```

## Usage

1.  **Download the Dataset:** Ensure you have the `fer2013.csv` file in your project directory or update the file path in the code.
2.  **Run the Script:** Execute the Python script. It will perform the following steps:
    *   Load and preprocess the data.
    *   Define and train a CNN model.
    *   Evaluate the model's performance on the test set.
    *   Visualize the training progress, predictions, and confusion matrix.
    *   Save the trained model.

## Code Explanation

The Python script `your_script_name.py` (replace with your actual script name) is organized into several sections:

*   **Import Libraries:** Imports necessary libraries for data manipulation, numerical operations, plotting, and building the neural network.
*   **Data Loading and Preparation:** Loads the FER2013 dataset, preprocesses the image data, and splits it into training, validation, and testing sets.
*   **Data Visualization:** Includes functions to visualize example images for each emotion and compare the distribution of emotions in different datasets.
*   **Model Definition:** Defines the architecture of the Convolutional Neural Network (CNN) model.
*   **Model Training and Evaluation:** Trains the defined CNN model on the training data and evaluates its performance on the test data.
*   **Results Visualization:** Visualizes the training loss and accuracy, individual predictions, and the confusion matrix to understand the model's performance.
*   **Model Saving:** Saves the trained model for future use.

## Key Files

*   `your_script_name.py`: The main Python script containing the code for data loading, preprocessing, model building, training, and evaluation.
*   `fer2013.csv`: The dataset file containing facial expression images and their corresponding emotion labels.

```python
# Importing necessary libraries
import pandas as pd # For data manipulation and analysis
import numpy as np # For numerical computations
import os # For interacting with the operating system
import matplotlib.pyplot as plt # For creating visualizations

# Importing functions for evaluating model performance
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix

# Importing layers and models from Keras for building the neural network
from keras import models
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import RMSprop, Adam

# Importing utility for one-hot encoding of labels
from tensorflow.keras.utils import to_categorical

# --- Data Loading and Initial Inspection ---
# Load the dataset from the specified CSV file
data = pd.read_csv(r'C:\Downloads\fer2013.csv')

# Display the last 5 entries of the 'Usage' column from row 1000 to 1020 (for a quick look)
data['Usage'][1000:1020].tail(5)

# Display the entire dataframe (for a comprehensive view of the loaded data)
data

# --- Data Preprocessing Function ---
# This function prepares the image data and labels from the dataframe
def prepare_data(data):
    # Initialize an empty numpy array to store image data
    image_array = np.zeros(shape=(len(data), 48, 48))
    # Convert emotion labels to a numpy array of integers
    image_label = np.array(list(map(int, data['emotion'])))

    # Iterate through each row of the dataframe
    for i, row in enumerate(data.index):
        # Convert the pixel string to an array of integers
        image = np.fromstring(data.loc[row, 'pixels'], dtype=int, sep=' ')
        # Reshape the pixel array into a 48x48 image
        image = np.reshape(image, (48, 48))
        # Store the reshaped image in the image_array
        image_array[i] = image

    # Return the processed image array and labels
    return image_array, image_label

# --- Visualization Functions ---
# Function to plot example images for a specific emotion
def plot_examples(label=0):
    # Create a figure and a set of subplots
    fig, axs = plt.subplots(1, 5, figsize=(25, 12))
    # Adjust spacing between subplots
    fig.subplots_adjust(hspace = .2, wspace=.2)
    # Flatten the array of axes for easy iteration
    axs = axs.ravel()
    # Loop through the first 5 images of the specified emotion
    for i in range(5):
        # Get the index of the i-th image with the given label
        idx = data[data['emotion']==label].index[i]
        # Display the image in grayscale
        axs[i].imshow(train_images[idx][:,:,0], cmap='gray')
        # Set the title of the subplot to the emotion name
        axs[i].set_title(emotions[train_labels[idx].argmax()])
        # Remove x and y axis ticks
        axs[i].set_xticklabels([])
        axs[i].set_yticklabels([])

# Function to plot one example image for each emotion
def plot_all_emotions():
    # Create a figure and a set of subplots for all 7 emotions
    fig, axs = plt.subplots(1, 7, figsize=(30, 12))
    # Adjust spacing between subplots
    fig.subplots_adjust(hspace = .2, wspace=.2)
    # Flatten the array of axes for easy iteration
    axs = axs.ravel()
    # Loop through each emotion
    for i in range(7):
        # Get the index of the first image with the current emotion label
        idx = data[data['emotion']==i].index[i]
        # Display the image in grayscale
        axs[i].imshow(train_images[idx][:,:,0], cmap='gray')
        # Set the title of the subplot to the emotion name
        axs[i].set_title(emotions[train_labels[idx].argmax()])
        # Remove x and y axis ticks
        axs[i].set_xticklabels([])
        axs[i].set_yticklabels([])

# Function to plot an image and its predicted emotion distribution
def plot_image_and_emotion(test_image_array, test_image_label, pred_test_labels, image_number):
    """ Function to plot the image and compare the prediction results with the label """

    # Create a figure and two subplots side by side
    fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharey=False)

    # Get the list of emotion names for the bar chart
    bar_label = emotions.values()

    # Display the image in the first subplot
    axs[0].imshow(test_image_array[image_number], 'gray')
    # Set the title of the first subplot to the actual emotion
    axs[0].set_title(emotions[test_image_label[image_number]])

    # Create a bar chart of the predicted probabilities in the second subplot
    axs[1].bar(bar_label, pred_test_labels[image_number], color='orange', alpha=0.7)
    # Add grid lines to the second subplot
    axs[1].grid()

    # Display the plot
    plt.show()

# Function to compare the distribution of emotions between two arrays of labels
def plot_compare_distributions(array1, array2, title1='', title2=''):
    # Create pandas DataFrames from the input arrays
    df_array1 = pd.DataFrame()
    df_array2 = pd.DataFrame()
    # Extract the emotion labels from the one-hot encoded arrays
    df_array1['emotion'] = array1.argmax(axis=1)
    df_array2['emotion'] = array2.argmax(axis=1)

    # Create a figure and two subplots side by side
    fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharey=False)
    # Get the list of emotion names
    x = emotions.values()

    # Count the occurrences of each emotion in the first array
    y = df_array1['emotion'].value_counts()
    # Handle cases where some emotions are missing in the array
    keys_missed = list(set(emotions.keys()).difference(set(y.keys())))
    for key_missed in keys_missed:
        y[key_missed] = 0
    # Create a bar chart for the first distribution
    axs[0].bar(x, y.sort_index(), color='orange')
    # Set the title of the first subplot
    axs[0].set_title(title1)
    # Add grid lines to the first subplot
    axs[0].grid()

    # Count the occurrences of each emotion in the second array
    y = df_array2['emotion'].value_counts()
    # Handle cases where some emotions are missing in the array
    keys_missed = list(set(emotions.keys()).difference(set(y.keys())))
    for key_missed in keys_missed:
        y[key_missed] = 0
    # Create a bar chart for the second distribution
    axs[1].bar(x, y.sort_index())
    # Set the title of the second subplot
    axs[1].set_title(title2)
    # Add grid lines to the second subplot
    axs[1].grid()

    # Display the plot
    plt.show()

# --- Data Exploration ---
# Count the occurrences of each usage type (Training, PublicTest, PrivateTest)
data['Usage'].value_counts()

# --- Emotion Label Mapping ---
# Define a dictionary to map emotion labels (integers) to emotion names
emotions = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

# --- Preparing Data for Training, Validation, and Testing ---
# Prepare image data and labels for the training set
train_image_array, train_image_label = prepare_data(data[data['Usage']=='Training'])
# Prepare image data and labels for the test set (PublicTest)
test_image_array, test_image_label = prepare_data(data[data['Usage']=='PublicTest'])
# Prepare image data and labels for the validation set (PrivateTest)
val_image_array, val_image_label = prepare_data(data[data['Usage']=='PrivateTest'])
#print(val_image_array) # Uncomment to inspect the validation image array

# Display the first image array from the training set
print(train_image_array[0])
#print(type(train_image_array[0][0][0])) # Uncomment to check the data type of pixel values

# --- Reshaping and Normalizing Images ---
# Reshape the training images to have a channel dimension (for CNN input) and normalize pixel values
train_images = train_image_array.reshape((train_image_array.shape[0], 48, 48, 1))
train_images = train_images.astype('float32')/255
# Reshape the validation images and normalize pixel values
val_images = val_image_array.reshape((val_image_array.shape[0], 48, 48, 1))
val_images = val_images.astype('float32')/255
# Reshape the test images and normalize pixel values
test_images = test_image_array.reshape((test_image_array.shape[0], 48, 48, 1))
test_images = test_images.astype('float32')/255
#print(val_images) # Uncomment to inspect the validation images after preprocessing
print(train_images[0][0],'\ntype: ', type(train_images[0][0][0][0]))

# --- One-Hot Encoding of Labels ---
# Convert training labels to one-hot encoded vectors
train_labels = to_categorical(train_image_label)
print(train_labels[0:15],'\n', train_image_label[0:15])
#print(train_labels.itemsize, train_image_label.itemsize )
