import tensorflow as tf
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error

from tensorflow.python.framework import dtypes
from tensorflow.contrib import layers as tflayers

# from data_processing import generate_data
# sudo pip3 install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.3.0-cp35-cp35m-linux_x86_64.whl


LOG_DIR = './ops_logs/sin'
TIMESTEPS = 3
RNN_LAYERS = [{'num_units': 5}]
DENSE_LAYERS = None
TRAINING_STEPS = 10000
PRINT_STEPS = TRAINING_STEPS / 10
BATCH_SIZE = 100
print('TensorFlow Version: ' + tf.__version__)

def lstm_model(num_units, rnn_layers, dense_layers=None, learning_rate=0.1, optimizer='Adagrad'):
    # num_units:        Size of the cells.
    # rnn_layers:       List of ints - The steps used to instantiate the `BasicLSTMCell` cell.
    # dense_layers:     List of nodes for each layer.
    #
    # Returns model definition

    def lstm_cells(layers):
        # layers:       List of ints.
        # num_units:    Number of units in the LSTM cell.
        #
        # Returns list of BasicLSTMCells??
        return [tf.contrib.rnn.BasicLSTMCell(num_units, state_is_tuple=True) for num_units in layers]

    def dnn_layers(input_layers, layers):
        if layers:
            # https://www.tensorflow.org/api_docs/python/tf/contrib/layers/stack
            return tflayers.stack(input_layers, tflayers.fully_connected, layers)
        else:
            return input_layers

    def _lstm_model(X, y):
        # https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/MultiRNNCell
        stacked_lstm = tf.contrib.rnn.MultiRNNCell(lstm_cells(rnn_layers), state_is_tuple=True)
        # unstack: Returns list of Tensor objects unstacked from X
        x_ = tf.unstack(X, axis=1, num=num_units)

        # static_rnn creates the recurrent neural network with a stacked_lstm cell.
        output, layers = tf.contrib.rnn.static_rnn(stacked_lstm, x_, dtype=dtypes.float32)
        output = dnn_layers(output[-1], dense_layers)

        # https://www.tensorflow.org/api_docs/python/tf/contrib/learn/models
        prediction, loss = learn.models.linear_regression(output, y)

        # https://www.tensorflow.org/api_docs/python/tf/contrib/layers/optimize_loss
        ###
        ### What is optimizer for?
        ###
        train_op = tf.contrib.layers.optimize_loss(loss, tf.contrib.framework.get_global_step(), optimizer=optimizer, learning_rate=learning_rate)
        return prediction, loss, train_op

    return _lstm_model

def rnn_data(data, time_steps, labels=False):
    """
    creates new data frame based on previous observation
      * example:
        l = [1, 2, 3, 4, 5]
        time_steps = 2
        -> labels == False [[1, 2], [2, 3], [3, 4]]
        -> labels == True [3, 4, 5]
    """
    rnn_df = []
    for i in range(len(data) - time_steps):
        if labels:
            try:
                rnn_df.append(data.iloc[i + time_steps].as_matrix())
            except AttributeError:
                rnn_df.append(data.iloc[i + time_steps])
        else:
            data_ = data.iloc[i: i + time_steps].as_matrix()
            rnn_df.append(data_ if len(data_.shape) > 1 else [[i] for i in data_])

    return np.array(rnn_df, dtype=np.float32)

def split_data(data, val_size=0.1, test_size=0.1):
    """
    splits data to training, validation and testing parts
    """
    ntest = int(round(len(data) * (1 - test_size)))
    nval = int(round(len(data.iloc[:ntest]) * (1 - val_size)))

    df_train, df_val, df_test = data.iloc[:nval], data.iloc[nval:ntest], data.iloc[ntest:]

    return df_train, df_val, df_test

def prepare_data(data, time_steps, labels=False, val_size=0.1, test_size=0.1):
    """
    Given the number of `time_steps` and some data,
    prepares training, validation and test data for an lstm cell.
    """
    df_train, df_val, df_test = split_data(data, val_size, test_size)
    return (rnn_data(df_train, time_steps, labels=labels),
            rnn_data(df_val, time_steps, labels=labels),
            rnn_data(df_test, time_steps, labels=labels))

def generate_data(fct, x, time_steps, seperate=True):
    """generates data with based on a function fct"""
    data = fct(x)
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    train_x, val_x, test_x = prepare_data(data['a'] if seperate else data, time_steps)
    train_y, val_y, test_y = prepare_data(data['b'] if seperate else data, time_steps, labels=True)
    return dict(train=train_x, val=val_x, test=test_x), dict(train=train_y, val=val_y, test=test_y)

def run_model():
    # Make the LSTM model with TIMESTEPS number of recurrent timesteps.
    make_model = lstm_model(TIMESTEPS, RNN_LAYERS, DENSE_LAYERS)
    
    # https://www.tensorflow.org/api_docs/python/tf/contrib/learn/Estimator
    # constructs an estimator instance.
    ###
    ### What is Estimator?
    ###
    make_estimator = tf.contrib.learn.Estimator(model_fn=make_model, model_dir=LOG_DIR)
    
    # https://www.tensorflow.org/api_docs/python/tf/contrib/learn/SKCompat
    # Returns a Scikit learn wrapper for TensorFlow Learn Estimator.
    ###
    ### What is this wrapper for? I think this is point of error because this is 
    ###     where numpy functions start being used instead of tf.
    ###
    regressor = tf.contrib.learn.SKCompat(make_estimator)

    # Generate some time-series sin data.
    # ----- switch to csv input.
    X, y = generate_data(np.sin, np.linspace(0, 100, 10000, dtype=np.float32), TIMESTEPS, seperate=False)

    # https://www.tensorflow.org/versions/r0.12/api_docs/python/contrib.learn.monitors/ops#ValidationMonitor.__init__
    # Create a validation monitor using validation data.
    # print(type(y['train']))
    # print(y['train'])
    input_fn = tf.contrib.learn.io.numpy_input_fn({"x": X['val']}, y['val'], num_epochs=TRAINING_STEPS)
    validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(input_fn=input_fn, every_n_steps=PRINT_STEPS, early_stopping_rounds=1000)
    #validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(x=X['val'], y=y['val'], every_n_steps=PRINT_STEPS, early_stopping_rounds=1000)

    # https://www.tensorflow.org/api_docs/python/tf/contrib/learn/SKCompat
    # ?Fit? the model using validation monitor and training data.
    # x['train'] is <class 'numpy.ndarray'>
    # y['train'] is <class 'numpy.ndarray'>
    fit_fn = tf.contrib.learn.io.numpy_input_fn({"x": X['train']}, y['train'], num_epochs=TRAINING_STEPS)
    regressor.fit(input_fn=fit_fn, monitors=[validation_monitor], batch_size=BATCH_SIZE, steps=TRAINING_STEPS)
    #regressor.fit(x=X['train'], y=y['train'], monitors=[validation_monitor], batch_size=BATCH_SIZE, steps=TRAINING_STEPS)
    
    # create predictions!
    predicted = regressor.predict(X['test'])

    # calculate root-mean-squared error.
    rmse = np.sqrt(((predicted - y['test']) ** 2).mean(axis=0))
    score = mean_squared_error(predicted, y['test'])
    print ("MSE: %f" % score)

run_model()




# def sin_cos(x):
#     return pd.DataFrame(dict(a=np.sin(x), b=np.cos(x)), index=x)
#
# X, y = generate_data(np.sin, np.linspace(0, 100, 10000, dtype=np.float32), TIMESTEPS, seperate=False)
# #print(prepare_data([.9,.5]))
# print(type(X))
# print(y)







#########################################
#########################################

# # -*- coding: utf-8 -*-
# from __future__ import absolute_import, division, print_function

# import numpy as np
# import pandas as pd


# def x_sin(x):
#     return x * np.sin(x)


# def sin_cos(x):
#     return pd.DataFrame(dict(a=np.sin(x), b=np.cos(x)), index=x)


# def rnn_data(data, time_steps, labels=False):
#     """
#     creates new data frame based on previous observation
#       * example:
#         l = [1, 2, 3, 4, 5]
#         time_steps = 2
#         -> labels == False [[1, 2], [2, 3], [3, 4]]
#         -> labels == True [3, 4, 5]
#     """
#     rnn_df = []
#     for i in range(len(data) - time_steps):
#         if labels:
#             try:
#                 rnn_df.append(data.iloc[i + time_steps].as_matrix())
#             except AttributeError:
#                 rnn_df.append(data.iloc[i + time_steps])
#         else:
#             data_ = data.iloc[i: i + time_steps].as_matrix()
#             rnn_df.append(data_ if len(data_.shape) > 1 else [[i] for i in data_])

#     return np.array(rnn_df, dtype=np.float32)


# def split_data(data, val_size=0.1, test_size=0.1):
#     """
#     splits data to training, validation and testing parts
#     """
#     ntest = int(round(len(data) * (1 - test_size)))
#     nval = int(round(len(data.iloc[:ntest]) * (1 - val_size)))

#     df_train, df_val, df_test = data.iloc[:nval], data.iloc[nval:ntest], data.iloc[ntest:]

#     return df_train, df_val, df_test


# def prepare_data(data, time_steps, labels=False, val_size=0.1, test_size=0.1):
#     """
#     Given the number of `time_steps` and some data,
#     prepares training, validation and test data for an lstm cell.
#     """
#     df_train, df_val, df_test = split_data(data, val_size, test_size)
#     return (rnn_data(df_train, time_steps, labels=labels),
#             rnn_data(df_val, time_steps, labels=labels),
#             rnn_data(df_test, time_steps, labels=labels))


# def load_csvdata(rawdata, time_steps, seperate=False):
#     data = rawdata
#     if not isinstance(data, pd.DataFrame):
#         data = pd.DataFrame(data)

#     train_x, val_x, test_x = prepare_data(data['a'] if seperate else data, time_steps)
#     train_y, val_y, test_y = prepare_data(data['b'] if seperate else data, time_steps, labels=True)
#     return dict(train=train_x, val=val_x, test=test_x), dict(train=train_y, val=val_y, test=test_y)


# def generate_data(fct, x, time_steps, seperate=False):
#     """generates data with based on a function fct"""
#     data = fct(x)
#     if not isinstance(data, pd.DataFrame):
#         data = pd.DataFrame(data)
#     train_x, val_x, test_x = prepare_data(data['a'] if seperate else data, time_steps)
#     train_y, val_y, test_y = prepare_data(data['b'] if seperate else data, time_steps, labels=True)
#     return dict(train=train_x, val=val_x, test=test_x), dict(train=train_y, val=val_y, test=test_y)