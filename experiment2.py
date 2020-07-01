import tensorflow as tf
from tensorflow import keras
import pandas as pd
import random
import numpy as np
from sklearn.preprocessing import minmax_scale

# Hyperparameters
LEARNING_RATE = 0.001
DROPOUT = 0.3
KERNEL_SIZE = 2
NUM_CONV1 = 32
NUM_CONV2 = 64
NUM_DENSE = 128

# Experimental setup
EPOCHS = 1
TRAINING_SIZE = 3000
TEST_SIZE = 1500
RANDOM_SIZE = 400

# Distributions that will be sampled from
distribution_dic = [
  'exponential', 
  'gamma',
#   'weibull'
]

# Function creating data samples. Parameters are step-wise increased.
# 'size' argument specifies the number of instances
# 'param' argument specifies the shape parameter of either the Gamma or Weibull distribution
def create_data(size, param):
    combined_data = []

    for j in range(0, len(distribution_dic)):
        labels = [0] * len(distribution_dic)
        labels[j] = 1
        for i in range(0, size):
            if (distribution_dic[j] == 'exponential'):
                rand_numbers = np.random.exponential(1, RANDOM_SIZE)
            if (distribution_dic[j] == 'gamma'):
                rand_numbers = np.random.gamma(param, 1, RANDOM_SIZE)
            # elif (distribution_dic[j] == 'weibull'):
            #     rand_numbers = np.random.weibull(1, RANDOM_SIZE)
    
            # Perform minmax normalization on the list of numbers
            rand_numbers = minmax_scale(rand_numbers)
            combined_data.append((rand_numbers, labels))
    
    # Shuffle data before splitting it in data set and labels set
    random.shuffle(combined_data)
    return combined_data

# Creates the CNN and processes the results.
# 'param' argument specifies the shape parameter of either the Gamma or Weibull distribution
def results(param):
    agg_data = create_data(TRAINING_SIZE, param)
    train_data = np.array([elem[0] for elem in agg_data])
    train_data = np.expand_dims(train_data, 2)
    train_labels = np.array([elem[1] for elem in agg_data])

    agg_test_data = create_data(TEST_SIZE, param)
    test_data = np.array([elem[0] for elem in agg_test_data])
    test_data = np.expand_dims(test_data, 2)
    test_labels = np.array([elem[1] for elem in agg_test_data])

    model = keras.models.Sequential()
    model.add(keras.layers.Conv1D(NUM_CONV1, KERNEL_SIZE, padding='same', activation='relu', input_shape=(RANDOM_SIZE,1)))
    model.add(keras.layers.Conv1D(NUM_CONV1, KERNEL_SIZE, padding='same', activation='relu'))
    model.add(keras.layers.MaxPooling1D(2, padding='valid'))
    model.add(keras.layers.Dropout(DROPOUT))
    model.add(keras.layers.Conv1D(NUM_CONV2, KERNEL_SIZE, padding='valid', activation='relu'))
    model.add(keras.layers.Conv1D(NUM_CONV2, KERNEL_SIZE, padding='valid', activation='relu'))
    model.add(keras.layers.MaxPooling1D(2, padding='valid'))
    model.add(keras.layers.Dropout(DROPOUT))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(NUM_DENSE, activation='relu'))
    model.add(keras.layers.Dense(len(distribution_dic), activation='softmax'))

    model.summary()

    optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)

    model.compile(optimizer=optimizer,
                    loss='categorical_crossentropy',
                    metrics=['categorical_accuracy'])

    model.fit(train_data, train_labels, epochs=EPOCHS)

    test_acc = model.evaluate(test_data,  test_labels)
    print('\nTest loss:', test_acc)

    predictions = model.predict(test_data)
    y_true = pd.Series([np.argmax(elem) for elem in test_labels])
    y_pred = pd.Series([np.argmax(elem) for elem in predictions])
    df_confusion = pd.crosstab(y_true, y_pred, rownames=['Actual'], colnames=['Predicted'])    
    df_columns = [('Predicted', elem) for elem in distribution_dic]
    try:
        df_confusion.columns = distribution_dic
        df_confusion.columns = pd.MultiIndex.from_tuples(df_columns)
        df_confusion.index = distribution_dic
    except:
        print('Wrong dimensions')
    df_confusion.index.name = 'Actual'
    print(df_confusion)
    with open('experiment2_results.csv','a') as fd:
        fd.write('\n\nWeibull param: ' + str(param) + '\n')
        fd.write('Test loss: ' + str(test_acc[0]) + '\n')
        fd.write('Test acc: ' + str(test_acc[1]) + '\n')
    df_confusion.to_csv('experiment2_results.csv', mode='a')

# Stepwise increase the parameter with 0.1 after every 5 runs
param = 1
for i in range(0,55):
    if i != 0 and i % 5 == 0:
        param = 1 + (i)/50
    results(param)
