# Alphabet Soup Charity - Neural Network Model

This project focuses on building and optimizing a deep learning model to predict the success of charity funding applications for the Alphabet Soup organization. The dataset includes various features related to the charity applications, and the goal is to create a binary classification model to determine whether an application will be successful.

## Project Overview

The purpose of this project was to develop a machine learning model capable of predicting the success rate of charity applications. The project involved data preprocessing, model building, and optimization using a neural network implemented with TensorFlow.

## Steps Completed

### 1. Data Preprocessing

- The dataset was preprocessed by removing non-beneficial columns (`EIN` and `NAME`).
- Categorical data was encoded using `pd.get_dummies()`.
- The dataset was split into training and testing sets.
- The features were scaled using `StandardScaler()` to prepare them for model training.

### 2. Compile, Train, and Evaluate the Model

- A neural network model was defined with an input layer, two hidden layers, and an output layer using TensorFlow and Keras.
- The model was compiled using a binary cross-entropy loss function and an Adam optimizer.
- The model was trained on the preprocessed training data, with early stopping callbacks to monitor performance.
- After training, the model was evaluated on the test data to determine its loss and accuracy.
- The initial model achieved an accuracy of approximately 73.26%.

### 3. Model Optimization

- Multiple attempts were made to optimize the model to reach a target accuracy above 75%.
- Adjustments included modifying the number of hidden layers, neurons, activation functions, and the number of epochs.
- The optimized model was saved and exported to an HDF5 file named `AlphabetSoupCharity_Optimization.h5`.

## Results

- **Loss:** 0.5552
- **Accuracy:** 73.26%

## Conclusion

The deep learning model successfully predicted the success of charity funding applications with reasonable accuracy. Although further optimization was performed, the model's final accuracy remained slightly below the target threshold of 75%. Future work could involve exploring different model architectures or using alternative machine learning techniques to further improve performance.

## Files Included

- **AlphabetSoupCharity.h5:** The saved model from the initial training.
- **AlphabetSoupCharity_Optimization.h5:** The saved model from the optimization phase.
- **charity_data.csv:** The dataset used for training and testing.
- **AlphabetSoupCharity_Optimization.ipynb:** The notebook containing the code for model optimization.
- **AlphabetSoupCharity.ipynb:** The notebook containing the code for data preprocessing, model building, and initial training.
