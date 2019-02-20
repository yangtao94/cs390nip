
# coding: utf-8

# In[4]:

from sklearn.metrics import confusion_matrix
from sklearn import datasets
from sklearn.model_selection import train_test_split
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import random
np.set_printoptions(threshold=np.inf)
# Setting random seeds to keep everything deterministic.
random.seed(1618)
np.random.seed(1618)
tf.set_random_seed(1618)

# Disable some troublesome logging.
tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Information on dataset.
NUM_CLASSES = 10
IMAGE_SIZE = 784

# Use these to set the algorithm to use.
#ALGORITHM = "guesser"
ALGORITHM = "custom_net"
#ALGORITHM = "tf_net"





class NeuralNetwork_2Layer():
    def __init__(self, inputSize, outputSize, neuronsPerLayer, learningRate = 0.01,addLayer = False,activation = "sigmoid"):
        #initialize neural network with option to add in 3rd layer, and to change the activation function
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.neuronsPerLayer = neuronsPerLayer
        self.lr = learningRate
        self.act = activation
        self.addLayer = addLayer
        if addLayer:
            self.W1 = np.random.randn(self.inputSize,self.neuronsPerLayer)
            self.W2 = np.random.randn(self.neuronsPerLayer,self.neuronsPerLayer)
            self.W3 = np.random.randn(self.neuronsPerLayer,self.outputSize)
        else:
            self.W1 = np.random.randn(self.inputSize, self.neuronsPerLayer)
            self.W2 = np.random.randn(self.neuronsPerLayer, self.outputSize)
    
    def activation_prime(self,x):
        if self.act == "sigmoid":
            return self.__sigmoidDerivative(x)
        elif self.act == "relu":
            return self.__reluDerivative(x)
        else:
            raise ValueError("Please select a valid activation function")
    def activation(self,x):
        if self.act == "sigmoid":
            return self.__sigmoid(x)
        elif self.act == "relu":
            return self.__relu(x)
        else:
            raise ValueError("Please select a valid activation function!")
    def __relu(self,x):
        for i in x:
            #print(i)
            np.maximum(i,0,i)
            #print(i)
        return x
    def __reluDerivative(self,x):
        #print("before",x[:1])
        for i in x:
            #print(i)
            i[i<=0] = 0
            i[i>0] = 1
            #print(i)
        #print("after",x[:1])
        return x
    
    
    # Activation function.
    def __sigmoid(self, x):
        return 1/(1+ np.exp(-x))

    # Activation prime function.
    def __sigmoidDerivative(self, x):
        return x*(1-x)

    # Batch generator for mini-batches. Not randomized.
    def __batchGenerator(self, l, n):
        for i in range(0, len(l), n):
            yield l[i : i + n]
    
    def backward(self, xVals, yVals,layers):
        
        d_o_errors = yVals - layers[-1]
        d_o_delta = d_o_errors*(self.activation_prime(layers[-1]))
        
        if(len(layers) == 3):
            d_w2_errors = d_o_delta.dot(self.W3.T)
            d_w2_delta = d_w2_errors*(self.activation_prime(layers[-2]))
        
        
            d_w1_errors = d_w2_delta.dot(self.W2.T)
            d_w1_delta = d_w1_errors*(self.activation_prime(layers[0]))
            self.W1 += self.lr * (xVals.T.dot(d_w1_delta))
            self.W2 += self.lr * (layers[0].T.dot(d_w2_delta))
            self.W3 += self.lr * (layers[1].T.dot(d_o_delta))
        
        #adjustment of the weights
        else:
            d_w2_errors = d_o_delta.dot(self.W2.T)
            d_w2_delta = d_w2_errors*(self.__activation_prime(layers[-2]))
            
            
            self.W1 += self.lr * (xVals.T.dot(d_w2_delta))
            self.W2 += self.lr * (layers[0].T.dot(d_o_delta))
        
        
    # Training with backpropagation.
    def train(self, xVals, yVals, epochs = 10, minibatches = True, mbs = 100):
        #print("Lakers",xVals.shape[1])
        y_length = yVals.shape[1]
        #TODO: Implement backprop. allow minibatches. mbs should specify the size of each minibatch.
        stuff = []
        #forward pass
        for i in range(epochs):
            #mini-batches
            print("iteration no: ",i)
            l = np.concatenate((xVals,yVals),axis=1)
            random.shuffle(l)
           
            #shuffle?
            get_batches = self.__batchGenerator(l,mbs)
            
            for value in get_batches:
                
                
                xVals_mini = value[:,:-y_length]
                yVals_mini = value[:,-y_length:]
                
                stuff = self.__forward(xVals_mini)
                #print("hihi",len(stuff))
                #backprop with minibatches
                self.backward(xVals_mini,yVals_mini,stuff)
        return stuff
    
     
        
        
    # Forward pass.
    def __forward(self, input):
        #print("dimensions",input.shape)
        #print("dimensions",self.W1.shape)
        
        net1 = np.dot(input,self.W1)
        layer1 = self.activation(net1)
        
        if self.addLayer is True:
            layer2 = self.activation(np.dot(layer1, self.W2))
            layer3 = self.activation(np.dot(layer2,self.W3))
            return layer1,layer2,layer3
        layer2 = self.activation(np.dot(layer1, self.W2))
        return layer1, layer2

    # Predict.
    def predict(self, xVals):
        #return only the output layer
        if self.addLayer is True:
            _,_,layer3 = self.__forward(xVals)
            return layer3
        
        _, layer2 = self.__forward(xVals)
        return layer2

    

# Classifier that just guesses the class label.
def guesserClassifier(xTest):
    ans = []
    for entry in xTest:
        pred = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        pred[random.randint(0, 9)] = 1
        ans.append(pred)
    return np.array(ans)




# In[ ]:




# In[5]:


#=========================<Pipeline Functions>==================================

def getRawData():
    mnist = tf.keras.datasets.mnist
    (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
    print("Shape of xTrain dataset: %s." % str(xTrain.shape))
    print("Shape of yTrain dataset: %s." % str(yTrain.shape))
    print("Shape of xTest dataset: %s." % str(xTest.shape))
    print("Shape of yTest dataset: %s." % str(yTest.shape))
    return ((xTrain, yTrain), (xTest, yTest))


# In[6]:

def preprocessData(raw):
    ((xTrain, yTrain), (xTest, yTest)) = raw            #TODO: Add range reduction here (0-255 ==> 0.0-1.0).
    xTrain = xTrain.reshape(60000,IMAGE_SIZE)
    xTest = xTest.reshape(10000,IMAGE_SIZE)
    #range deduction
    denom = 255
    xTrain = xTrain / denom
    xTest = xTest/denom
    yTrainP = to_categorical(yTrain, NUM_CLASSES)
    yTestP = to_categorical(yTest, NUM_CLASSES)
    print("New shape of xTrain dataset: %s." % str(xTrain.shape))
    print("New shape of xTest dataset: %s." % str(xTest.shape))
    print("New shape of yTrain dataset: %s." % str(yTrainP.shape))
    print("New shape of yTest dataset: %s." % str(yTestP.shape))
    return ((xTrain, yTrainP), (xTest, yTestP))


# In[7]:

def getRawData_iris():
    iris = datasets.load_iris()
    attributes = iris.data
    labels = iris.target
    xTrain,xTest,yTrain,yTest = train_test_split(attributes,labels,test_size = 0.5)
    return ((xTrain,yTrain),(xTest,yTest))
def preprocess_iris(raw):
    ((xTrain,yTrain),(xTest,yTest)) = raw
    yTrainP = to_categorical(yTrain,3)
    yTestP = to_categorical(yTest,3)
    print("New shape of xTrain dataset: %s." % str(xTrain.shape))
    print("New shape of xTest dataset: %s." % str(xTest.shape))
    print("New shape of yTrain dataset: %s." % str(yTrainP.shape))
    print("New shape of yTest dataset: %s." % str(yTestP.shape))
    return ((xTrain,yTrainP),(xTest,yTestP))


# In[35]:



def trainModel(data):
    xTrain, yTrain = data
    if ALGORITHM == "guesser":
        return None   # Guesser has no model, as it is just guessing.
    elif ALGORITHM == "custom_net":
        print("Building and training Custom_NN.")
        activation = ["sigmoid","relu"]    
        #TODO: Write code to build and train your custon neural net.
        #initialize network
        #set add layer to True
        custom_net = NeuralNetwork_2Layer(xTrain.shape[1],yTrain.shape[1],30,0.1,True,activation[0])
        
        #Change batch size to 1 when testing the iris dataset
        weights = custom_net.train(xTrain,yTrain,20,True,128)
        print("trained!")
        return custom_net
    elif ALGORITHM == "tf_net":
        print("Building and training TF_NN.")
        #TODO: Write code to build and train your keras neural net.
        model = tf.keras.Sequential()
        lossType = tf.keras.losses.categorical_crossentropy
        opt = tf.train.AdamOptimizer()
        i_shape = (784,)
        model.add(tf.keras.layers.Dense(512, input_shape = i_shape,activation = tf.nn.relu))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Dense(512,activation=tf.nn.relu))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Dense(10,activation = tf.nn.softmax))
        model.compile(optimizer = opt, loss = lossType)
        model.fit(xTrain,yTrain,batch_size = 128,epochs = 10)
        return model
    else:
        raise ValueError("Algorithm not recognized.")



# In[36]:


def runModel(data, model):
    if ALGORITHM == "guesser":
        return guesserClassifier(data)
    elif ALGORITHM == "custom_net":
        print("Testing Custom_NN.")
        #print("Not yet implemented.")                   #TODO: Write code to run your custon neural net.
        pred = model.predict(data)
        
        return pred
    elif ALGORITHM == "tf_net":
        print("Testing TF_NN.")
        #TODO: Write code to run your keras neural net.
        preds = model.predict(data)
        return preds
    else:
        raise ValueError("Algorithm not recognized.")


# In[37]:

def evalResults(data, preds):   #TODO: Add F1 score confusion matrix here.
    xTest, yTest = data
    acc = 0
    y_pred = []
    y_true = []
    for i in range(preds.shape[0]):
        y_pred.append(np.argmax(preds[i],0))
        y_true.append(np.argmax(yTest[i],0))
        if (np.argmax(preds[i],0) == np.argmax(yTest[i],0)):   acc = acc + 1
    accuracy = acc / preds.shape[0]
    #Print confusion matrix from sklearn
    array = confusion_matrix(y_true,y_pred)
    print(array)
    print("Classifier algorithm: %s" % ALGORITHM)
    print("Classifier accuracy: %f%%" % (accuracy * 100))
    print()


# In[38]:

#=========================<Main>================================================

def main():
    ##Uncomment when trying to test the iris_dataset
    #change batch size to one when training the model
#    if ALGORITHM != "tf_net":
#        raw = getRawData_iris()
#        data = preprocess_iris(raw)
#        model = trainModel(data[0])
#        preds = runModel(data[1][0],model)
#        evalResults(data[1],preds)
    
    raw = getRawData()
    data = preprocessData(raw)
    model = trainModel(data[0])
    preds = runModel(data[1][0], model)
    evalResults(data[1], preds)



if __name__ == '__main__':
    main()


# In[ ]:




# In[ ]:




# In[ ]:



