# Alphabet Soup Charity - Neural Network Model

This project focuses on building and optimizing a deep learning model to predict the success of charity funding applications for the Alphabet Soup organization. The dataset includes various features related to the charity applications, and the goal is to create a binary classification model to determine whether an application will be successful.

(The phrase "Alphabet Soup Charity - Neural Network Model" typically refers to a project or analysis involving a neural network model applied to data from a charity organization named "Alphabet Soup.")

## Project Overview

The purpose of this project was to develop a machine learning model capable of predicting the success rate of charity applications. The project involved data preprocessing, model building, and optimization using a neural network implemented with TensorFlow.

- Neural Network Model: This indicates the type of machine learning model being used. A neural network model is a computational model inspired by the human brain’s network of neurons. It consists of layers of interconnected nodes (neurons) that process data through various transformations to learn patterns and make predictions.

## Steps Completed
### Dependencies
![image](https://github.com/user-attachments/assets/f8cca40c-475c-4f53-a87f-9dd44c2f97c0)

### 1. Data Preprocessing

- The dataset was preprocessed by removing non-beneficial columns (`EIN` and `NAME`).
  ![image](https://github.com/user-attachments/assets/de87c05d-b88a-49e0-a3f7-d3dd331853b1)
  - ![image](https://github.com/user-attachments/assets/078844b0-d44b-4ecc-b8f7-ab82bac549a6)

- Categorical data was encoded using `pd.get_dummies()`.
- The dataset was split into training and testing sets.
- ![image](https://github.com/user-attachments/assets/1453db19-e71b-4d95-b663-3c6cb1c62607)

- The features were scaled using `StandardScaler()` to prepare them for model training.

### 2. Compile, Train, and Evaluate the Model

- A neural network model was defined with an input layer, two hidden layers, and an output layer using TensorFlow and Keras.
  
![image](https://github.com/user-attachments/assets/739f20d4-c4e2-4e5e-aee9-677eacc50cc6)

- The model was compiled using a binary cross-entropy loss function and an Adam optimizer.
- The model was trained on the preprocessed training data, with early stopping callbacks to monitor performance.
- After training, the model was evaluated on the test data to determine its loss and accuracy.
- The initial model achieved an accuracy of approximately 73.26%.
- ![image](https://github.com/user-attachments/assets/f6f35e1c-941d-450e-8389-06554bde0269)


### 3. Model Optimization

- Multiple attempts were made to optimize the model to reach a target accuracy above 75%.
- Adjustments included modifying the number of hidden layers, neurons, activation functions, and the number of epochs.
- The optimized model was saved and exported to an HDF5 file named `AlphabetSoupCharity_Optimization.h5`.
- Model : - **[AlphabetSoupCharity_Optimization.h5](./AlphabetSoupCharity_Optimization.h5):** The saved model from the optimization phase.

## Results

- **Loss:** 0.5552
- **Accuracy:** 73.26%

![image](https://github.com/user-attachments/assets/ea49c5a0-564c-4771-86c2-498a075b0e48)


## Conclusion

### Report - [Neural Network Model Report.pdf](./Neural%20Network%20Model%20Report.pdf)


The deep learning model successfully predicted the success of charity funding applications with reasonable accuracy. Although further optimization was performed, the model's final accuracy remained slightly below the target threshold of 75%. Future work could involve exploring different model architectures or using alternative machine learning techniques to further improve performance.

## Files Included

- **AlphabetSoupCharity.h5:** The saved model from the initial training.
- **AlphabetSoupCharity_Optimization.h5:** The saved model from the optimization phase.
- **charity_data.csv:** The dataset used for training and testing.
- **AlphabetSoupCharity_Optimization.ipynb:** The notebook containing the code for model optimization.
- **AlphabetSoupCharity.ipynb:** The notebook containing the code for data preprocessing, model building, and initial training.
