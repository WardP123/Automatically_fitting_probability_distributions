import tensorflow as tf
from tensorflow import keras
import pandas as pd
import random
import numpy as np
from sklearn.preprocessing import minmax_scale
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorboard.plugins.hparams import api as hp

# Experimental setup
EPOCHS = 1
TRAINING_SIZE = 2000
TEST_SIZE = 500

# Varying feature set length that is sampled from for the hyperparameter
# optimisation
RANDOM_SIZE =  [50,100,200,400,800,1600]

# Parameters for generating artificially generated samples, 
# chosen to not confuse the classifier with similar distributions
BINOM = 0.001
ALPHA = 0.5
BETA = 0.5
N = 20
NGOOD = 1
NBAD = 50

METRIC_ACCURACY = 'categorical_accuracy'

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
def create_data(size):
    combined_data = []

    for j in range(0, len(distribution_dic)):
        labels = [0] * len(distribution_dic)
        labels[j] = 1
        for i in range(0, size):
            length = random.sample(RANDOM_SIZE, 1)
            if (distribution_dic[j] == 'beta'):
                rand_numbers = np.random.beta(ALPHA, BETA, length)
            elif (distribution_dic[j] == 'binomial'):
                rand_numbers = np.random.binomial(N, BINOM, length)
            elif (distribution_dic[j] == 'chisquare'):
                rand_numbers = np.random.chisquare(0.1, length)
            elif (distribution_dic[j] == 'exponential'):
                rand_numbers = np.random.exponential(0.001, length)
            elif (distribution_dic[j] == 'gamma'):
                rand_numbers = np.random.gamma(10, 1, length)
            elif (distribution_dic[j] == 'geometric'):
                rand_numbers = np.random.geometric(1, length)
            elif (distribution_dic[j] == 'hypergeometric'):
                rand_numbers = np.random.hypergeometric(NGOOD, NBAD, N, length)
            elif (distribution_dic[j] == 'lognormal'):
                rand_numbers = np.random.lognormal(0, 1, length)
            elif (distribution_dic[j] == 'normal'):
                rand_numbers = np.random.normal(0, 1, length)
            elif (distribution_dic[j] == 'poisson'):
                rand_numbers = np.random.poisson(1, length)
            elif (distribution_dic[j] == 'standard_t'):
                rand_numbers = np.random.standard_t(1, length)
            elif (distribution_dic[j] == 'uniform'):
                rand_numbers = np.random.uniform(0, 1, length)
            elif (distribution_dic[j] == 'weibull'):
                rand_numbers = np.random.weibull(0.01, length)

            # Perform minmax normalization on the list of numbers
            rand_numbers = minmax_scale(rand_numbers)
            # Pad the shorter feature set length with the mean to give every instance
            # an equal feature set length
            rand_numbers = np.pad(rand_numbers, int(1/2 * (max(RANDOM_SIZE) - len(rand_numbers))), 'mean')
            combined_data.append((rand_numbers, labels))

    # Shuffle data before splitting it in data set and labels set
    random.shuffle(combined_data)
    return combined_data

# Creates the training data
agg_data = create_data(TRAINING_SIZE)
train_data = np.array([elem[0] for elem in agg_data])
train_data = np.expand_dims(train_data, 2)
train_labels = np.array([elem[1] for elem in agg_data])

# Creates the testing data
agg_test_data = create_data(TEST_SIZE)
test_data = np.array([elem[0] for elem in agg_test_data])
test_data = np.expand_dims(test_data, 2)
test_labels = np.array([elem[1] for elem in agg_test_data])

# Specify the hyperparameters. Multiple values for the hyperparameters can be 
# passed in the array for  grid-search. Individual hyperparameters can be 
# intuitively adapted for random-search. 
HP_NUM_CONV1 = hp.HParam('num_conv1', hp.Discrete([32]))
HP_NUM_CONV2 = hp.HParam('num_conv2', hp.Discrete([64]))
HP_DROPOUT = hp.HParam('dropout', hp.Discrete([0.3]))
HP_KERNEL_SIZE = hp.HParam('kernel_size', hp.Discrete([1]))
HP_NUM_DENSE = hp.HParam('num_dense', hp.Discrete([128]))
HP_LEARNING_RATE = hp.HParam('learning_rate', hp.Discrete([0.001]))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam']))

#Setting the Metric to RMSE
METRIC_CA = 'categorical_accuracy'

#Creating & configuring log files
with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
  hp.hparams_config(
    hparams=[HP_NUM_DENSE, HP_NUM_CONV1, HP_NUM_CONV2, HP_DROPOUT, HP_KERNEL_SIZE, 
    HP_LEARNING_RATE, HP_OPTIMIZER],
    metrics=[hp.Metric(METRIC_CA, display_name='Categorical_accuracy')]
)

# Creates the CNN
def train_test_model(hparams):
    model = keras.models.Sequential()
    model.add(keras.layers.Conv1D(hparams[HP_NUM_CONV1], hparams[HP_KERNEL_SIZE], padding='same', activation='relu', input_shape=(max(RANDOM_SIZE),1)))
    model.add(keras.layers.Conv1D(hparams[HP_NUM_CONV1], hparams[HP_KERNEL_SIZE], padding='same', activation='relu'))
    model.add(keras.layers.MaxPooling1D(2, padding='valid'))
    model.add(keras.layers.Dropout(hparams[HP_DROPOUT]))
    model.add(keras.layers.Conv1D(hparams[HP_NUM_CONV2], hparams[HP_KERNEL_SIZE], padding='valid', activation='relu'))
    model.add(keras.layers.Conv1D(hparams[HP_NUM_CONV2], hparams[HP_KERNEL_SIZE], padding='valid', activation='relu'))
    model.add(keras.layers.MaxPooling1D(2, padding='valid'))
    model.add(keras.layers.Dropout(hparams[HP_DROPOUT]))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(hparams[HP_NUM_DENSE], activation='relu'))
    model.add(keras.layers.Dense(len(distribution_dic), activation='softmax'))

    model.summary()

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(optimizer=hparams[HP_OPTIMIZER],
                    loss='categorical_crossentropy',
                    metrics=['categorical_accuracy'])
    model.fit(
        train_data, 
        train_labels, 
        epochs=EPOCHS,
    )
    return model.evaluate(test_data,  test_labels)

# Runs the hyperparameter optimisation
def run(run_dir, hparams):
    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)  # record the values used in this trial
        accuracy = train_test_model(hparams)[1]
        tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)


session_num = 0

# Creates the setup for the hyperparameter optimisation
for num_conv1 in HP_NUM_CONV1.domain.values:
    for num_conv2 in HP_NUM_CONV2.domain.values:
        for dropout_rate in (HP_DROPOUT.domain.values):
            for kernel_size in (HP_KERNEL_SIZE.domain.values):
                for learning_rate in (HP_LEARNING_RATE.domain.values):
                    for num_dense in HP_NUM_DENSE.domain.values:
                        for optimizer in HP_OPTIMIZER.domain.values:
                            hparams = {
                                HP_NUM_CONV1: num_conv1,
                                HP_NUM_CONV2: num_conv2,
                                HP_DROPOUT: dropout_rate,
                                HP_KERNEL_SIZE: kernel_size,
                                HP_LEARNING_RATE: learning_rate,
                                HP_NUM_DENSE: num_dense,
                                HP_OPTIMIZER: optimizer,
                            }
                            run_name = "run-%d" % session_num
                            print('--- Starting trial: %s' % run_name)
                            print({h.name: hparams[h] for h in hparams})
                            run('logs/hparam_tuning/' + run_name, hparams)
                            session_num += 1