# Neural Network - Alphabet Soup Charity Analysis

This project focuses on building and optimizing a deep learning model to predict the success of charity funding applications for the Alphabet Soup organization. The dataset includes various features related to the charity applications, and the goal is to create a binary classification model to determine whether an application will be successful.

(The phrase "Alphabet Soup Charity - Neural Network Model" typically refers to a project or analysis involving a neural network model applied to data from a charity organization named "Alphabet Soup.")

## Project Overview

With our knowledge of machine learning and neural networks, we used the features in the provided dataset to create a binary classifier that is capable of predicting whether applicants will be successful if funded by Alphabet Soup.

The purpose of this project was to develop a machine learning model capable of predicting the success rate of charity applications. The project involved data preprocessing, model building, and optimization using a neural network implemented with TensorFlow.

From Alphabet Soup’s business team, we received a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization, such as the following:
  
![image](https://github.com/user-attachments/assets/d91afd61-f7c8-4023-afc0-663669ae93df)

- Neural Network Model: This indicates the type of machine learning model being used. A neural network model is a computational model inspired by the human brain’s network of neurons. It consists of layers of interconnected nodes (neurons) that process data through various transformations to learn patterns and make predictions.

## Technologies Used:
- TensorFlow
- Keras and Keras-Tuner
- Scikit-learn
- Pandas
- Jupyter notebook
- Google Colaboratory
- MatplotLib and PyPlot

## Steps Completed

### 1. Data Preprocessing

- What variable(s) are considered the target(s) for your model?
- ![image](https://github.com/user-attachments/assets/97e4137c-0f83-443f-9f88-034a487bb5cf)

- What variable(s) are considered to be the features for your model?
- ![image](https://github.com/user-attachments/assets/042f98fe-4f83-4d8c-96d9-336e1a3f6fd3)

- The dataset was preprocessed by removing non-beneficial columns (`EIN` and `NAME`).
  ![image](https://github.com/user-attachments/assets/c0418537-39fe-42ea-8fc2-b5c54f08610c)
  ![image](https://github.com/user-attachments/assets/6bddc617-b6a0-4d88-a302-199853adba21)

- Categorical data was encoded using `pd.get_dummies()`.
- The dataset was split into training and testing sets.
  
- ![image](https://github.com/user-attachments/assets/259eaf4d-8fa4-48f2-b79c-97ea889ae095)
- The features were scaled using `StandardScaler()` to prepare them for model training.
  
- ![image](https://github.com/user-attachments/assets/e2fd7140-c35b-4fec-b148-2dcdae9b461f)

### 2. Compile, Train, and Evaluate the Model

- How many neurons, layers, and activation functions did you select for your neural network model, and why?
- ![image](https://github.com/user-attachments/assets/31e15153-5a53-4c7d-94c9-1fd5b988f4c1)

- A neural network model was defined with an input layer, two hidden layers, and an output layer using TensorFlow and Keras.
  
![image](https://github.com/user-attachments/assets/46f85934-ea4c-4541-adcb-59b9dbd9ef30)

- The model was compiled using a binary cross-entropy loss function and an Adam optimizer.
- The model was trained on the preprocessed training data, with early stopping callbacks to monitor performance.
- ![image](https://github.com/user-attachments/assets/bdd62d4c-d3f9-4431-9f0d-3201fc5aa35d)

- What steps did you take to try and increase model performance?
- I tried just about everything:

   - Reduced the number of unique values for several features (APPLICATION_TYPE, CLASSIFICATION).
   - ![image](https://github.com/user-attachments/assets/b87a342d-4864-4de6-a0f3-42a732c31f2a)
   - ![image](https://github.com/user-attachments/assets/4c86b3d8-4655-4a70-80c7-d9c681fd0dd2)
   - Converted numeric ranges presented as strings (INCOME_AMT), back into integer values.
   - Binned integer values (ASK_AMT) back into numeric ranges presented as strings (so that they could be one-hot encoded)
   - Scaled values.
   - Used Keras Tuner to automate the selection of HyperParameters.
   - Increased the number of epochs used in training to very high values.
   - The most important step was putting the Name column back into the features, mapping names to integers, and binning them.
- After training, the model was evaluated on the test data to determine its loss and accuracy.
- The initial model achieved an accuracy of approximately 73.26%.
- ![image](https://github.com/user-attachments/assets/198b006c-cbf9-448b-bcfe-02dcb07c5ba1)



### 3. Model Optimization

- Multiple attempts were made to optimize the model to reach a target accuracy above 75%.
- Adjustments included modifying the number of hidden layers, neurons, activation functions, and the number of epochs.
- The optimized model was saved and exported to an HDF5 file named `AlphabetSoupCharity_Optimization.h5`.
- Model : - **[AlphabetSoupCharity_Optimization.h5](./AlphabetSoupCharity_Optimization.h5):** The saved model from the optimization phase.

## Results

- **Loss:** 0.5552
- **Accuracy:** 73.26%

![image](https://github.com/user-attachments/assets/ea49c5a0-564c-4771-86c2-498a075b0e48)

# Graphs
![image](https://github.com/user-attachments/assets/7ec7c067-925d-4dbd-b1ae-6af175663baa)  
![image](https://github.com/user-attachments/assets/ce4fdca9-49da-4967-aa7d-4e88edc7cff5)  
![image](https://github.com/user-attachments/assets/5bd6ad96-8520-48fd-b003-5475ab891a7d)  
![image](https://github.com/user-attachments/assets/27f35a7f-4c48-4bab-9353-e017d9255276)  
![image](https://github.com/user-attachments/assets/612186fb-0f44-4221-839a-944480f6c281)  
![image](https://github.com/user-attachments/assets/3b673a55-b80f-4af3-8319-80b7a6a03327)  
![image](https://github.com/user-attachments/assets/887831f7-7247-4a62-adc2-bdfc24fb41f1)  








## Conclusion

### Report - [Neural Network Model Report.pdf](./Neural%20Network%20Model%20Report.pdf)


The deep learning model successfully predicted the success of charity funding applications with reasonable accuracy. Although further optimization was performed, the model's final accuracy remained slightly below the target threshold of 75%. Future work could involve exploring different model architectures or using alternative machine learning techniques to further improve performance.

## Files Included

- **AlphabetSoupCharity.h5:** The saved model from the initial training.
- **AlphabetSoupCharity_Optimization.h5:** The saved model from the optimization phase.
- **charity_data.csv:** The dataset used for training and testing.
- **AlphabetSoupCharity_Optimization.ipynb:** The notebook containing the code for model optimization.
- **AlphabetSoupCharity.ipynb:** The notebook containing the code for data preprocessing, model building, and initial training.
