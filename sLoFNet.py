# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 00:37:26 2023

@author: sajad
"""

import tensorflow as tf
import tensorflow.keras as keras
from numpy import *
from matplotlib.pyplot import *
from tensorflow.keras import backend as K
from hyperopt import *

class sLoFNet:
    """ A class for the singe Lorentzian Fitting Neural Network (sLoFNet)
    
    Attributes
    ----------
    x_values : container type
        The x values of the curves focused on
    nr_of_points : int
        Number of discrete values in the curve
    nr_of_hidden_nodes : int
        The number of nodes in the hidden layers
    nr_of_layers : int
        Number of hidden layers in the network
    lr : float
        learning rate for at the time of initialization
    optimized : Bool or keras.Model object
        False if no hyperparameter optimization has been done. Otherwise the
        a keras.Model object of the optimized model configuration
    batch_size : int
        The batch size used to train the neural network. [32, 64, 128, 256]
    loss : function
        Loss function used to train neural newtork
    model_dense : keras.Model object
        The manually specificed model configuration 
        
    Methods
    -------
    simulate_data(values, nr_of_instances, noise_lvl = 0.0097)
        Simulates random lorentzian shapes and returns the sample with its
        corresponding labels.
    
    optimize()
        Uses the hyperopt library to search for an optimal model configuration.
        Generates a keras.Model object as the attribute optimized to replace the
        keras.Model object stored as the attribute model_dense
        
    train(training_samples, training_labels, epochs = 100, reduction_scheme = True, history = False)
        Trains either model_dense or optimized (if it has been generated)
    
    fit(samples)
        Predicts the shape parameters of the input
    """
    
    def __init__(self, x_values, nr_of_points=26, nr_of_hidden_layers = 5, nr_of_hidden_nodes = 2048,final_activation = None,loss = None, batch_size = 32, lr = 0.001):
        """
        Parameters
        ----------
        x_values : container type
            specify the x values of the targeted Lorentzians
        nr_of_points : int
            The number of x-values
        nr_of_hidden_layers : int
            number of hidden layers
        nr_of_hidden_nodes : int
            number of nodes in the hidden layers
        final_activation : str
            The activation function used at the final layer. "final" or "tanh"
        loss : function
            The loss function used for training the network
        batch_size : int
            The number of samples given to the network during training during
            each iteration
        lr : float
            The learning rate used during the initialization of training
        """
        self.x_values = x_values
        self.nr_of_points = nr_of_points
        self.nr_of_hidden_nodes = nr_of_hidden_nodes
        self.nr_of_layers = nr_of_hidden_layers
        self.lr = lr
        self.optimized = False
        self.batch_size = batch_size
        
        if loss:
            self.loss = loss
        else:
            self.loss = lambda y_true,y_pred: K.sqrt(K.mean(K.square(y_pred - y_true)))
        
        self.model_dense = keras.Sequential()
        self.model_dense.add(keras.layers.Input(nr_of_points))
        for i in range(nr_of_hidden_layers):
            self.model_dense.add(keras.layers.Dense(nr_of_hidden_nodes, kernel_initializer="normal", activation="relu"))
        self.model_dense.add(keras.layers.Dense(3, kernel_initializer="normal", activation=final_activation))
    
    def __repr__(self):
        return f"Single Lorentzian Fitting Neural Networkand for input with {self.nr_of_points} points"
    
    def __reduction_scheme(self,model, training_samples, training_labels, loss = "default", lr_start = 0.001,batch_size = None, history= False):
        early_stop = keras.callbacks.EarlyStopping(monitor = "loss", patience = 2)
        h= model.fit(x=training_samples,y=training_labels,validation_split = .1, epochs = 10, batch_size = batch_size, callbacks= [early_stop],shuffle = True, verbose = 0)
        hi = dict()
        hi["start"] = h
        lr = lr_start
        if loss == "default":
            while lr>1e-9:
                lr/=10
                model.compile(keras.optimizers.Adam(lr), loss = self.loss, metrics = ["accuracy"])
                h2= model.fit(training_samples,training_labels,validation_split = .1, epochs = 10,batch_size = batch_size, callbacks= [early_stop],shuffle = True, verbose = 0)
                hi[f"{lr}"]= h2
        else:
            while lr>1e-9:
                lr/=10
                model.compile(keras.optimizers.Adam(lr), loss = loss, metrics = ["accuracy"])
                h2= model.fit(training_samples,training_labels,validation_split = .1, epochs = 10,batch_size = batch_size, callbacks= [early_stop],shuffle = True, verbose = 0)
                hi[f"{lr}"]= h2
            
        if history:
            return hi 
        
    def simulate_data(self, nr_of_instances, noise_lvl = 0.0097 ):
        """Parameters
        ----------
        nr_of_instances : int
            Number of sample-label pairs to simulate
       
        noise_lvl : float
            The noise level on the simulated samples
        
        """
        lorentzian = lambda deltaf_rf,A = .8,LW=.1,deltaf_H20 = .05: A*(1-((LW**2)/(LW**2+4*(deltaf_rf-deltaf_H20)**2)))+(1-A)
        samples = []
        labels = []
        for i in range(nr_of_instances):
            A = random.uniform(.3,1.)
            LW = random.uniform(10/128,160/128)
            deltaf_h20 = random.uniform(-1,1)
            a = sqrt((lorentzian(self.x_values,A,LW,deltaf_h20)+np.random.normal(scale = noise_lvl, size = self.x_values.shape[0]))**2+np.random.normal(scale = noise_lvl,size =self.x_values.shape[0])**2)
            samples.append(a/np.max(a))
            labels.append([A,LW,deltaf_h20])
        samples = np.asarray(samples)
        labels = np.asarray(labels)
        return samples, labels
    
    def optimize(self):
        train_samples,train_labels = self.simulate_data(10000)
        test_samples,test_labels = self.simulate_data(10000)
        # Model function
        def model_d1(inp = self.nr_of_points,loss = self.loss, nr_of_layers=None,nr_of_nodes = None, final_activation = "linear"):
            model_dense = keras.Sequential()
            model_dense.add(keras.layers.Input(inp))
    
            for i in range(nr_of_layers):
                model_dense.add(keras.layers.Dense(nr_of_nodes,kernel_initializer="normal", activation="relu"))
        
            model_dense.add(keras.layers.Dense(3, kernel_initializer="normal", activation=final_activation))
    
 
            model_dense.compile(optimizer="adam", loss = loss)        
            return model_dense
        
        ##### Step 1. Definition of Objective function
        # Looks at the combined loss seen to Mean absolute error and Mean squared error    
        def objective_function(args):
            
            model = model_d1(loss = args["loss"],nr_of_layers = args["nr_of_layers"],nr_of_nodes = args["nr_of_nodes"], final_activation=
                             args["final_activation"])
            
            h = self.__reduction_scheme(model, train_samples,train_labels, loss = args["loss"], batch_size =args["batch_size"])
            
            p = model.predict(test_samples, verbose = 0)
            
            mae = mean(keras.losses.MAE(p,test_labels))
            mse = mean(keras.losses.MSE(p,test_labels))
            return mae+mse
            
        ##### Step 2. Definition of search space
        #paramters to include:
            # Number of hidden layers: 1-10 
            # Number of nodes per layer: 64,128,256,512,1024 (there was a paper talking about more features is more valueable than depth)
            # Final activation function: tanh or linear (linear prefarred here)
            # loss function: rmse or mae

        space = {"loss":hp.choice("loss",(self.loss,"mae")),"nr_of_layers":hp.choice("nr_of_layers",(1,2,3,4,5,6,7,8,9,10)),
                 "nr_of_nodes":hp.choice("nr_of_nodes",(128,256,512,1028,2048)), 
                 "final_activation":hp.pchoice("final_activation",[(.8,"linear"),(.2,"tanh")]),
                 "batch_size":hp.choice("batch_size",(32,64,128,256))}     

        ##### Step 3. Choisce of serach algorithm and running the search
        best = fmin(objective_function, space, algo = rand.suggest, max_evals = 5)
        model_config = space_eval(space, best)
        model_config_s = [f"{i}: {model_config[i]}" for i in model_config.keys()]
        print(f"\nSuccessful optimization yielded model with the following configuration: \n\n{model_config_s[0]} \n{model_config_s[1]} \n{model_config_s[2]} \n{model_config_s[3]} \n{model_config_s[4]}")
        
        self.optimized = model_d1(self.nr_of_points, loss = model_config["loss"], nr_of_layers=model_config["nr_of_layers"],nr_of_nodes = model_config["nr_of_nodes"], final_activation = model_config["final_activation"])
        self.batch_size = model_config["batch_size"]
        
    def train(self, training_samples, training_labels, epochs = 100, reduction_scheme = True, history = False):
        """
        Parameters
        ----------
        training_samples : numpy array
            samples to train on
        
        training_labels : numpy array
            corresponding labels
            
        epochs : int
            Number of epochs to traing the network
        
        reduction_scheme : bool
            Whether or not to reduce the lr during training to accelerate convergence
        
        history : bool
            Whether or not to return the history of training and validation losses
        """
        if self.optimized:
            model = self.optimized
            model.compile(optimizer="adam", loss = self.loss)
        else: 
            model = self.model_dense
            model.compile(optimizer="adam", loss = self.loss)
            
        if reduction_scheme:
            early_stop = keras.callbacks.EarlyStopping(monitor = "loss", patience = 1)
            h= model.fit(x=training_samples,y=training_labels,validation_split = .1, epochs = epochs, batch_size = self.batch_size, callbacks= [early_stop],shuffle = True)
            hi = dict()
            hi["start"] = h
            lr = self.lr
            
            while lr>1e-6:
                lr/=10
                model.compile(keras.optimizers.Adam(lr), loss = self.loss)
                h2= model.fit(training_samples,training_labels,validation_split = .1, epochs = 50,batch_size = self.batch_size, callbacks= [early_stop],shuffle = True)
                hi[f"{lr}"]= h2
                
            if history:
                return hi
        else:
            h= model.fit(x=training_samples,y=training_labels,validation_split = .1, epochs = epochs, batch_size = batch_size, callbacks= [early_stop],shuffle = True)
            return h

        print(f"\nTraining on {len(training_samples)} samples for {epochs} epochs successfully completed")
          
       
    def fit(self,samples):
        """
        Parameters 
        ----------
        samples : numpy array
            samples to predict the shape paramters for
        """
        if self.optimized:
            return self.optimized.predict(samples.reshape(-1,self.nr_of_points))
        else:
            return self.model_dense.predict(samples.reshape(-1,self.nr_of_points)) 
        
