"""
LSTM for time series classification

This model takes in time series and class labels.
The LSTM models the time series. A fully-connected layer
generates an output to be classified with Softmax
"""

import numpy as np
import tensorflow as tf  #TF 1.1.0rc1
tf.logging.set_verbosity(tf.logging.ERROR)
import matplotlib.pyplot as plt
from LSTM.LSTM_Model import Model,sample_batch,load_data#,check_test




#Set these directories
class control_LSTM(object):
    def __init__(self, *args):
        """ Example on how to use class below under __name__ is __main__ stmt
        """
        try:
            self.direc = """prodData"""
            self.summaries_dir = 'LSTM_LOG/log_tb'
            self.data = dict()
            self.configPrep = dict()
            self.configLSTM = None
            self.epochs = None
        except:
            self.failureMessage(0)
            quit()

    def failureMessage(self, errorCode):
        """ Continue to build upon this function whenever there areas that can fail.
        """
        def quittingM():
            print("=== control_LSTM says | Now quitting. Goodbye.")

        if errorCode == 0:
            print("=== control_LSTM says | Failure to initialize due to input. Not enough args, perhaps.")
            quittingM()
        elif errorCode == 1:
            print("""=== control_LSTM says | Failure to load data. You probably forgot
                    to place the data directory within the current working dir. """)
            quittingM()

    def initData(self, *args):
        print("=== control_LSTM says | initializing data...")
        try:
            """Load the data"""
            ratio = np.array([0.8,0.9]) #Ratios where to split the training and validation set
            X_train,X_val,X_test,y_train,y_val,y_test = load_data(self.direc,ratio,dataset='powerGeneratedWindTurbine')
            self.data['X_train'] = X_train
            self.data['X_val'] = X_val
            self.data['X_test'] = X_test
            self.data['y_train'] = y_train
            self.data['y_val'] = y_val
            self.data['y_test'] = y_test

            N,sl = X_train.shape
            self.configPrep['N'] = N
            self.configPrep['sl'] = sl

            #print("***** N, SL: ", N,sl)
            num_classes = len(np.unique(y_train))
            self.configPrep['num_classes'] = num_classes
            print("Done loading data.")
        except:
            self.failureMessage(1)
            quit()

    def configureLSTM(self, batch_size=300, max_iterations=1000, dropout=0.8, num_layers=3, hidden_size=120, max_grad_norm=3, mu=0.005 ):
        """Hyperparamaters"""
        print("=== control_LSTM says | configuring lstm...")
        self.data['batch_size'] = batch_size
        self.configPrep['max_iterations'] = max_iterations
        self.configPrep['dropout'] = dropout
        self.configLSTM = {    'num_layers' :    num_layers,     #number of layers of stacked RNN's
                      'hidden_size' :   hidden_size,             #memory cells in a layer
                      'max_grad_norm' : max_grad_norm,           #maximum gradient norm during training
                      'batch_size' :    self.data['batch_size'],
                      'learning_rate' : mu,
                      'sl':             self.configPrep['sl'],
                      'num_classes':    self.configPrep['num_classes']}

        self.configLSTM['epochs']  = np.floor(self.data['batch_size']*self.configPrep['max_iterations'] / self.configPrep['N'])
        print('Train %.0f samples in approximately %d epochs' %(self.configPrep['N'],self.configLSTM['epochs']))

    def saveModel(self, session, targetFile):
        raise NotImplementedError

    def buildModel(self, *args):
        print("=== control_LSTM says | building new model...")
        #Instantiate a model
        model = Model(self.configLSTM)

        """Session time"""
        sess = tf.Session() #Depending on your use, do not forget to close the session
        writer = tf.summary.FileWriter(self.summaries_dir, sess.graph)  #writer for Tensorboard
        sess.run(model.init_op)

        cost_train_ma = -np.log(1/float(self.configPrep['num_classes'])+1e-9)  #Moving average training cost
        acc_train_ma = 0.0
        try:
          for i in range(self.configPrep['max_iterations']):
            X_batch, y_batch = sample_batch(self.data['X_train'],self.data['y_train'], self.data['batch_size'])
            #print("X_batch: ", X_batch)
            #print("y_batch: ", y_batch)
            #Next line does the actual training
            # print("model.cost: ", model.cost)
            # print("model.accuracy: ", model.accuracy)
            # print("model.train_op: ", model.train_op)
            # print("*****")
            # print("model.input: ", model.input)
            # print("X_batch: ", X_batch)
            # print("model.labels: ", model.labels)
            # print("y_batch: ", y_batch)
            #print(sess.run([model.cost,model.accuracy, model.train_op],feed_dict = {model.input: X_batch,model.labels: y_batch,model.keep_prob:dropout}))
            #z=raw_input()
            cost_train, acc_train,_ = sess.run([model.cost,model.accuracy, model.train_op],feed_dict = {model.input: X_batch,model.labels: y_batch,model.keep_prob:self.configPrep['dropout']})
            #print("cost_train: ", cost_train)
            #print("acc_train: ", acc_train)
            #z=input()
            cost_train_ma = cost_train_ma*0.99 + cost_train*0.01
            acc_train_ma = acc_train_ma*0.99 + acc_train*0.01
            if i%100 == 1:
            #Evaluate validation performance
              X_batch, y_batch = sample_batch(self.data['X_val'],self.data['y_val'],self.configLSTM['batch_size'])
              cost_val, summ,acc_val = sess.run([model.cost,model.merged,model.accuracy],feed_dict = {model.input: X_batch, model.labels: y_batch, model.keep_prob:1.0})
              print('At %5.0f/%5.0f: COST %5.3f/%5.3f(%5.3f) -- Acc %5.3f/%5.3f(%5.3f)' %(i,self.configPrep['max_iterations'],cost_train,cost_val,cost_train_ma,acc_train,acc_val,acc_train_ma))
              #Write information to TensorBoard
              writer.add_summary(summ, i)
              writer.flush()
        except KeyboardInterrupt:
            print("***** control_LSTM says: KeyboardInterrupt, or somethin' funky occured. ")
            pass

        epoch = float(i)*self.data['batch_size']/self.configPrep['N']
        print('Trained %.1f epochs, accuracy is %5.3f and cost is %5.3f'%(epoch,acc_val,cost_val))


        fileToSaveModel = "????"
        self.saveModel(sess, fileToSaveModel)
        print("Model saved to ...."+fileToSaveModel)



    def runModel(self, modelTarget, dataStreamObj):
        """ (1) Run a specific model and pair it with a stream of data
            for online applications.
            (2) if no modelTarget specified, or none exist, create one.
        """
        def loadModel(modelTarget):
            """ returns a sess object
            """
            raise NotImplementedError

        def displayExistingModels():
            """ prints to screen what models exist
                and some info on their attributes
                and returns a list of model names
                ** Also give options if no models exist.
            """
            raise NotImplementedError

        ready = False
        usrIn = None
        options = []

        while not (ready):
            if modelTarget is None:
                options = displayExistingModels()
                usrIn = input()
            if usrIn in options:
                modelTarget = usrIn
                ready = True

        if modelTarget == "create":
            print("=== control_LSTM says | Beginning creation of new model...")
            self.initAndBuildModel()

        else:
            # TODO begin running the model
            sess = loadModel()

            while (True):
                # continue to run model until user interrupt
                print("=== control_LSTM says | TODO: implement an existing model here.")
        quit()


    def initAndBuildModel(self, *args):
        """ initializes the model and starts it.
        """
        self.initData()
        self.configureLSTM() # batch_size=300, max_iterations=1000, dropout=0.8, num_layers=3, hidden_size=120, max_grad_norm=3, mu=0.005 )
        self.buildModel()

if __name__ == "__main__":
    freshLstm = control_LSTM("direc Where Data Is", "Where to log files")
    lstmConfigOptions = {batch_size: 300, max_iterations: 1000, dropout: 0.8, num_layers: 3, hidden_size: 120, max_grad_norm:3, mu:0.005 }
    freshLstm.initAndBuildModel(lstmConfigOptions)
#now run in your terminal:
# $ tensorboard --logdir = <summaries_dir>
# Replace <summaries_dir> with your own dir
