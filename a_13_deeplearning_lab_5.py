# -*- coding: utf-8 -*-
"""A_13_DeepLearning_Lab_5.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1lVJeTQ6nIqGuGgAlaiKkN1LovhCW4jCt

# **Deep Learning - Lab 5**

**Name :** Radhika Shukla

**Batch :** A1

**Roll No.:** 13

**Aim :** Design and develop an Artificial Neural Network model using below conditions for diabetes prediction.
        
1. Perform EDA·        
2. Train and evaluate the performance of model using appropriate metrics.
3. Display run time chart
4. Apply regularization technique like L1, L2, dropout and early stopping to improve the performance of the model.
5. Experiment with individual or combination of regularization techniques.
6. Provide comparative analysis
7. Save the ANN model


Tips:

L1 and L2:
from tensorflow.keras.regularizers import l2,l1
Model.add(Dense(8,activation='relu',kernel_regularizer=l2(l2=0.1)))



Early Stooping:
from tensorflow.keras.callbacks import EarlyStopping
callback = EarlyStopping(monitor="val_loss", min_delta=0.00001,patience=20,verbose=1,mode="auto",baseline=None, restore_best_weights=False)


irisModel.fit(x_train,y_train_encoded,epochs=1000,validation_data=(x_test,y_test_encoded),callbacks=callback)

Dropout


irisModel2.add(Dense(8,activation='relu'))
irisModel2.add(Dropout(0.4))
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import seaborn as sns
from tensorflow.keras.callbacks import EarlyStopping
import time
from tensorflow.keras.regularizers import l2,l1
from tensorflow.keras.layers import Dropout

df = pd.read_excel('/content/diabetes.xlsx')

df.head()

"""### Checking for null values"""

df.isnull().sum()

"""### Checking for duplicate values"""

df[df.duplicated]

plt.figure(figsize=(14,8))
sns.heatmap(df.corr(numeric_only=True),annot=True)

df.columns

"""### Applying Train-Test-Split"""

X = df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age']]
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""### Applying Standard Scaler"""

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

"""### Creating an ANN Model"""



model = Sequential([
    Dense(128, activation='relu', input_shape=(8,)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

start_time = time.time()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=200, batch_size=32, validation_data=(X_test, y_test))
end_time = time.time()

model.save('diabetic_ann_.h5')

"""### Applying Regularization techniques

#### Using L1, L2 regularization
"""

callback = EarlyStopping(monitor="val_loss", min_delta=0.00001,patience=20,verbose=1,mode="auto",baseline=None, restore_best_weights=False)

model = Sequential([
    Dense(128, activation='relu', input_shape=(8,), kernel_regularizer=l1(0.01)),
    Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

start_time_l1l2 = time.time()
model.fit(X_train, y_train, epochs=200, batch_size=32, validation_data=(X_test, y_test))
end_time_l1l2 = time.time()

model.save('diabetic_ann_l1l2.h5')

"""#### Using Early Stopping"""

model = Sequential([
    Dense(128, activation='relu', input_shape=(8,)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

start_time_earlystop = time.time()
model.fit(X_train, y_train, epochs=200, batch_size=32, validation_data=(X_test, y_test), callbacks=callback)
end_time_earlystop = time.time()

model.save('diabetic_ann_earlystopping.h5')

"""#### Using L1, L2 regularization, EarlyStopping, and Dropout"""

model = Sequential([
    Dense(128, activation='relu', input_shape=(8,), kernel_regularizer=l1(0.01)),
    Dropout(0.4),
    Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.4),
    Dense(32, activation='relu'),
    Dropout(0.4),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

start_time_regularized_all = time.time()
model.fit(X_train, y_train, epochs=200, batch_size=32, validation_data=(X_test, y_test), callbacks=callback)
end_time_regularized_all = time.time()

model.save('diabetic_ann_regularized_all.h5')

"""### Calculating Runtime of model

#### Without Regularization
"""

print("Training time: ", end_time - start_time)

plt.plot([start_time, end_time], [0,1])
plt.title('Model Training Runtime (without regularization)')
plt.ylabel('Progress')
plt.xlabel('Time')
plt.show()

"""#### Using L1 and L2 Regularization"""

print("Training time: ", end_time_l1l2 - start_time_l1l2)

plt.plot([start_time_l1l2, end_time_l1l2], [0,1])
plt.title('Model Training Runtime (L1L2 regularization)')
plt.ylabel('Progress')
plt.xlabel('Time')
plt.show()

"""#### Using EarlyStopping Regularization"""

print("Training time: ", end_time_earlystop - start_time_earlystop)

plt.plot([start_time_earlystop, end_time_earlystop], [0,1])
plt.title('Model Training Runtime (EarlyStopping Regularization)')
plt.ylabel('Progress')
plt.xlabel('Time')
plt.show()

"""#### Using L1L2, Dropout, and EarlyStopping Regularization"""

print("Training time: ", end_time_regularized_all - start_time_regularized_all)

plt.plot([start_time_regularized_all, end_time_regularized_all], [0,1])
plt.title('Model Training Runtime (L1L2, EarlyStopping, and Dropout Regularization)')
plt.ylabel('Progress')
plt.xlabel('Time')
plt.show()

"""---

### Calculating the Accuracy of model using test dataset
"""

accuracy = model.evaluate(X_test, y_test)[1]
print("Accuracy: ", accuracy)

