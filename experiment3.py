import tensorflow as tf
from tensorflow import keras
import pandas as pd
import random
import numpy as np
from sklearn.preprocessing import minmax_scale
from scipy import stats

# Hyperparameters
LEARNING_RATE = 0.001
DROPOUT = 0.3
KERNEL_SIZE = 2
NUM_CONV1 = 32
NUM_CONV2 = 64
NUM_DENSE = 128

# Experimental setup
EPOCHS = 1
TRAINING_SIZE = 2000
TEST_SIZE = 1000
RANDOM_SIZE = 800

distribution_dic = [
  'binomial', 
  'exponential', 
  'normal',
  'weibull',
]

# Uses Maximum Likelihood Estimation to calculate the parameters based
# on real-world data
# 'dist' argument specifies which distribution to calculate parameters from
# 'sample_data' argument gives the data from which the parameters are caculated
def calc_parameters(dist, sample_data):
    if (dist == 'Binomial'):
        trial = sample_data[0]
        numb_success = sample_data.count(trial)
        success = numb_success/len(sample_data)
        return success
    elif (dist == 'Exponential'):
        return stats.expon.fit(sample_data)[0]
    elif (dist == 'Normal'):
        return stats.norm.fit(sample_data)
    elif (dist == 'Weibull'):
        return stats.exponweib.fit(sample_data)[1]

# Function creating data samples. Parameters are step-wise increased.
# 'size' argument specifies the number of instances
# 'param' argument specifies the shape parameter of either the Gamma or Weibull distribution
def create_data(size, data):
    combined_data = []
    obj = {}

    # Calculate the parameters once
    obj['Binomial'] = calc_parameters('Binomial', data)
    obj['Exponential'] = calc_parameters('Exponential', data)
    obj['Normal'] = calc_parameters('Normal', data)
    obj['Weibull'] = calc_parameters('Weibull', data)

    for j in range(0, len(distribution_dic)):
        labels = [0] * len(distribution_dic)
        labels[j] = 1
        for i in range(0, size):
            if (distribution_dic[j] == 'binomial'):
                params = obj['Binomial']
                rand_numbers = np.random.binomial(1000, params, RANDOM_SIZE)
            elif (distribution_dic[j] == 'exponential'):
                params = obj['Exponential']
                rand_numbers = np.random.exponential(params, RANDOM_SIZE)
            elif (distribution_dic[j] == 'normal'):
                params = obj['Normal']
                mean = params[0]
                std = params[1]
                rand_numbers = np.random.normal(mean, std, RANDOM_SIZE)
            elif (distribution_dic[j] == 'weibull'):
                params = obj['Weibull']
                rand_numbers = np.random.weibull(params, RANDOM_SIZE)

            # Perform minmax normalization on the list of numbers
            rand_numbers = minmax_scale(rand_numbers)
            combined_data.append((rand_numbers, labels))
    
    # Shuffle data before splitting it in data set and labels set
    random.shuffle(combined_data)
    return combined_data

# Creates the CNN and processes the results.
# 'sample_data' argument passes the real-world data
# 'name' argument specifies the name of the distribution
# 'results' argument specifies the labels per distribution
def get_results(sample_data, name, results):
    data = np.random.choice(minmax_scale(sample_data), int(len(sample_data)/10)).tolist()

    agg_train_data = create_data(TRAINING_SIZE, data)
    train_data = np.array([elem[0] for elem in agg_train_data])
    train_data = np.expand_dims(train_data, 2)
    train_labels = np.array([elem[1] for elem in agg_train_data])

    test_data = []
    test_labels = []
    for i in range(0,TEST_SIZE):
        test_data.append(np.array(minmax_scale(sample_data.sample(n=RANDOM_SIZE, replace=True).tolist())))
        sample_data_labels = np.array(results)
        test_labels.append(sample_data_labels)
    test_data = np.expand_dims(np.array(test_data), 2)
    test_labels = np.array(test_labels)

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
    with open('experiment3_results.csv','a') as fd:
        fd.write('\n\nData: ' + str(name) + '\n')
        fd.write('Test loss: ' + str(test_acc[0]) + '\n')
        fd.write('Test acc: ' + str(test_acc[1]) + '\n')
    df_confusion.to_csv('experiment3_results.csv', mode='a')

# Main loop to train and test CNN multiple times for different data sets
for i in range(0,4):
    for j in range(0,5):
        print(i,j)
        if i == 0:
            df_coin = pd.read_csv('./Data/unbiased coins toss.csv')
            df_coin = df_coin.replace('H', 0)
            df_coin = df_coin.replace('T', 1)
            coin = df_coin['1_re_old'].append(df_coin['1_re_new'])
            coin = coin.append(df_coin['2_rs_old'])
            coin = coin.append(df_coin['2_rs_new'])
            coin = coin.append(df_coin['5_rs_old'])
            coin = coin.append(df_coin['5_rs_new'])
            sample_data = coin
            name = 'Binomial'
            results = [1,0,0,0]
        elif i == 1:
            df_claim = pd.read_csv('./Data/Inpatient_Claim.csv')
            claim = df_claim['CLM_PMT_AMT']
            sample_data = claim
            name = 'Exponential'
            results = [0,1,0,0]
        elif i == 2:
            df_height = pd.read_csv('./Data/athlete_events.csv')
            height = df_height['Height'].dropna()
            sample_data = height
            name = 'Normal'
            results = [0,0,1,0]
        elif i == 3:
            df_fail = pd.read_csv('./Data/aps_failure_test_set.csv')
            fail = df_fail['ad_000']
            fail = fail[fail != 'na']
            sample_data = fail
            name = 'Weibull'
            results = [0,0,0,1]
        get_results(sample_data, name, results)