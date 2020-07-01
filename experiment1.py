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
EPOCHS = 50
TRAINING_SIZE = 2000
TEST_SIZE = 1000

# Varying feature set lengths
RANDOM_SIZE = [
    1600, 1600, 1600, 1600, 1600,
    800, 800, 800, 800, 800,
    400, 400, 400, 400, 400,
    200, 200, 200, 200, 200, 
    100, 100, 100, 100, 100,
    50, 50, 50, 50, 50
]

# Parameters for generating artificially generated samples, 
# chosen to not confuse the classifier with similar distributions
BINOM = 0.01
ALPHA = 0.5
BETA = 0.5
N = 20
NGOOD = 1
NBAD = 50

# All distributions that will be sampled from
distribution_dic = [
  'beta', 
  'binomial', 
  'chisquare', 
  'exponential', 
  'gamma',
  'geometric',
  'hypergeometric',
  'lognormal',
  'normal',
  'poisson',
  'standard_t',
  'uniform',
  'weibull',
]

# Function creating data samples. Parameters are pre-determined. 
# 'size' argument specifies the number of instances
# 'rand_size' argument specifies the feature set length
def create_data(size, rand_size):
    combined_data = []

    for j in range(0, len(distribution_dic)):
        labels = [0] * len(distribution_dic)
        labels[j] = 1
        for i in range(0, size):
            if (distribution_dic[j] == 'beta'):
                rand_numbers = np.random.beta(ALPHA, BETA, rand_size)
            elif (distribution_dic[j] == 'binomial'):
                rand_numbers = np.random.binomial(N, BINOM, rand_size)
            elif (distribution_dic[j] == 'chisquare'):
                rand_numbers = np.random.chisquare(0.1, rand_size)
            elif (distribution_dic[j] == 'exponential'):
                rand_numbers = np.random.exponential(0.001, rand_size)
            elif (distribution_dic[j] == 'gamma'):
                rand_numbers = np.random.gamma(10, 1, rand_size)
            elif (distribution_dic[j] == 'geometric'):
                rand_numbers = np.random.geometric(1, rand_size)
            elif (distribution_dic[j] == 'hypergeometric'):
                rand_numbers = np.random.hypergeometric(NGOOD, NBAD, N, rand_size)
            elif (distribution_dic[j] == 'lognormal'):
                rand_numbers = np.random.lognormal(0, 1, rand_size)
            elif (distribution_dic[j] == 'normal'):
                rand_numbers = np.random.normal(0, 1, rand_size)
            elif (distribution_dic[j] == 'poisson'):
                rand_numbers = np.random.poisson(1, rand_size)
            elif (distribution_dic[j] == 'standard_t'):
                rand_numbers = np.random.standard_t(1, rand_size)
            elif (distribution_dic[j] == 'uniform'):
                rand_numbers = np.random.uniform(0, 1, rand_size)
            elif (distribution_dic[j] == 'weibull'):
                rand_numbers = np.random.weibull(0.01, rand_size)

            # Perform minmax normalization on the list of numbers
            rand_numbers = minmax_scale(rand_numbers)
            combined_data.append((rand_numbers, labels))

    # Shuffle data before splitting it in data set and labels set
    random.shuffle(combined_data)
    return combined_data

# Creates the CNN and processes the results.
# 'rand_size' argument specifies the feature set length
def results(rand_size):
    agg_data = create_data(TRAINING_SIZE, rand_size)
    train_data = np.array([elem[0] for elem in agg_data])
    train_data = np.expand_dims(train_data, 2)
    train_labels = np.array([elem[1] for elem in agg_data])

    agg_test_data = create_data(TEST_SIZE, rand_size)
    test_data = np.array([elem[0] for elem in agg_test_data])
    test_data = np.expand_dims(test_data, 2)
    test_labels = np.array([elem[1] for elem in agg_test_data])

    model = keras.models.Sequential()
    model.add(keras.layers.Conv1D(NUM_CONV1, KERNEL_SIZE, padding='same', activation='relu', input_shape=(rand_size,1)))
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
    with open('experiment1_results.csv','a') as fd:
        fd.write('\n\nSize: ' + str(rand_size) + '\n')
        fd.write('Test loss: ' + str(test_acc[0]) + '\n')
        fd.write('Test acc: ' + str(test_acc[1]) + '\n')
    df_confusion.to_csv('experiment1_results.csv', mode='a')

# Calls the results function for every feature set length
for length in RANDOM_SIZE:
    results(length)
