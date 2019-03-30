
# coding: utf-8

# In[27]:


import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
import random
from scipy.misc import imsave, imresize
from scipy.optimize import fmin_l_bfgs_b   # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_l_bfgs_b.html
from tensorflow.keras.applications import vgg19
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import warnings
import matplotlib.pyplot as plt
import time

random.seed(1618)
np.random.seed(1618)
tf.set_random_seed(1618)

tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

CONTENT_IMG_PATH = "ben-simmons.jpg"           #TODO: Add this.
STYLE_IMG_PATH = "mona_lisa.jpg"             #TODO: Add this.


CONTENT_IMG_H = 500
CONTENT_IMG_W = 500

STYLE_IMG_H = 500
STYLE_IMG_W = 500

CONTENT_WEIGHT = 0.1    # Alpha weight.
STYLE_WEIGHT = 5.0      # Beta weight.
TOTAL_WEIGHT = 0.5

TRANSFER_ROUNDS = 10

#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

#=============================<Helper Fuctions>=================================
'''
TODO: implement this.
This function should take the tensor and re-convert it to an image.
'''
def deprocessImage(img):
    img = img.reshape((CONTENT_IMG_H),(CONTENT_IMG_W),3)
    #add back mean
    img[:,:,0] += 103.939
    img[:,:,1] += 116.779
    img[:,:,2] += 123.68
    #reverse order of colors
    img = img[:,:,::-1]
    img = np.clip(img,0,255).astype('uint8')
    return img


def gramMatrix(x):
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram








# In[28]:

#========================<Loss Function Builder Functions>======================

def styleLoss(style, gen):
    return K.sum(K.square(gramMatrix(style)-gramMatrix(gen)))/(4.* (3**2)*(CONTENT_IMG_H * CONTENT_IMG_W)**2)


def contentLoss(content, gen):
    return K.sum(K.square(gen - content))


def totalLoss(x):
    #reduce noise
    a = K.square(x[:,:CONTENT_IMG_H-1,:CONTENT_IMG_W-1,:] - x[:,1:,:CONTENT_IMG_W-1,:])
    b = K.square(x[:,:CONTENT_IMG_H-1,:CONTENT_IMG_W-1,:] - x[:,:CONTENT_IMG_H-1,1:,:])
    
    
    return K.sum(K.pow(a+b,1.25))   #TODO: implement.



# In[29]:


#=========================<Pipeline Functions>==================================

def getRawData():
    print("   Loading images.")
    print("      Content image URL:  \"%s\"." % CONTENT_IMG_PATH)
    print("      Style image URL:    \"%s\"." % STYLE_IMG_PATH)
    cImg = load_img(CONTENT_IMG_PATH)
   
    
    
    tImg = cImg.copy()
    sImg = load_img(STYLE_IMG_PATH)
    
    y = img_to_array(sImg)
    
    #plt.imshow(np.uint8(y))
    #plt.show()
    
    print("      Images have been loaded.")
    return ((cImg, CONTENT_IMG_H, CONTENT_IMG_W), (sImg, STYLE_IMG_H, STYLE_IMG_W), (tImg, CONTENT_IMG_H, CONTENT_IMG_W))



def preprocessData(raw):
    img, ih, iw = raw
    img = img_to_array(img)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        img = imresize(img, (ih, iw, 3))
    img = img.astype("float64")
    #plt.imshow(np.uint8(img))
    #plt.show()
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img


# In[30]:

def eval_loss_and_grads(x):
    x = x.reshape((1,CONTENT_IMG_H,CONTENT_IMG_W,3))
    #print(x)
    output = f_outputs([x])
    loss_value = output[0]
    
    if len(output[1:]) == 1:
        grad_values = output[1].flatten().astype('float64')
    else:
        grad_values = np.array(output[1:]).flatten().astype('float64')
    return loss_value,grad_values



class Evaluator(object):
    def __init__(self):
        self.loss_value = None
        self.grads_values = None
    
    def loss(self,x):
        #assert self.loss_value is None
        loss_value,grad_values = eval_loss_and_grads(x)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value
    
    def grads(self,x):
        #assert self.grad_values is None
        temp = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return temp

       

evaluator = Evaluator()
f_outputs = None
x = None
'''
TODO: Allot of stuff needs to be implemented in this function.
First, make sure the model is set up properly.
Then construct the loss function (from content and style loss).
Gradient functions will also need to be created, or you can use K.Gradients().
Finally, do the style transfer with gradient descent.
Save the newly generated and deprocessed images.
'''
def styleTransfer(cData, sData, tData):
    print("   Building transfer model.")
    contentTensor = K.variable(cData)
    styleTensor = K.variable(sData)
    genTensor = K.placeholder((1, CONTENT_IMG_H, CONTENT_IMG_W, 3))
    inputTensor = K.concatenate([contentTensor, styleTensor, genTensor], axis=0)
    model = vgg19.VGG19(include_top=False, weights = "imagenet", input_tensor=inputTensor)   #TODO: implement.
    outputDict = dict([(layer.name, layer.output) for layer in model.layers])
    print("   VGG19 model loaded.")
    loss = 0.0
    styleLayerNames = ["block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1", "block5_conv1"]
    contentLayerName = "block5_conv2"
    print("   Calculating content loss.")
    contentLayer = outputDict[contentLayerName]
    contentOutput = contentLayer[0, :, :, :]
    genOutput = contentLayer[2, :, :, :]
    loss += CONTENT_WEIGHT * contentLoss(contentOutput,genOutput)   #TODO: implement.
    print("   Calculating style loss.")
    for layerName in styleLayerNames:
        styleLayer = outputDict[layerName]
        styleOutput = styleLayer[1,:,:,:]
        genOutput = styleLayer[2,:,:,:]
        loss += (STYLE_WEIGHT/len(styleLayerNames)) * styleLoss(styleOutput,genOutput)   #TODO: implement.
    loss += TOTAL_WEIGHT * totalLoss(genTensor)   #TODO: implement.
    # TODO: Setup gradients or use K.gradients().
    gradient = K.gradients(loss,genTensor)
    #create K.function to output loss and gradients
    print(type(gradient))
    outputs = [loss]
    outputs += gradient
    global f_outputs
    global x
    f_outputs = K.function([genTensor],outputs)
    x = cData
    print("   Beginning transfer.")
    for i in range(TRANSFER_ROUNDS):
        print("   Step %d." % i)
        #TODO: perform gradient descent using fmin_l_bfgs_b.
        start_time = time.time()
        x,tLoss,ph = fmin_l_bfgs_b(evaluator.loss,x.flatten(),fprime = evaluator.grads,maxfun = 20)
        
        print("      Loss: %f." % tLoss)
        img = deprocessImage(x.copy())
        filename = 'hello'+str(i)
        saveFile = filename + '.jpg'   #TODO: Implement.
        imsave(saveFile, img)   #Uncomment when everything is working right.
        end_time = time.time()
        print("      Image saved to \"%s\"." % saveFile)
        print('Iteration %d completed in %ds' % (i, end_time - start_time))
    
    
#    print("   Beginning transfer.")
#    for i in range(TRANSFER_ROUNDS):
#        print("   Step %d." % i)
        #TODO: perform gradient descent using fmin_l_bfgs_b.
#        
#        x,tLoss,ph = fmin_l_bfgs_b(evaluator.loss,x.flatten(),fprime = evaluator.grads,maxiter=100)
        
#        print("      Loss: %f." % tLoss)
#        img = deprocessImage(x)
#        saveFile = 'hello.jpg'   #TODO: Implement.
#        imsave(saveFile, img)   #Uncomment when everything is working right.
#        print("      Image saved to \"%s\"." % saveFile)
 #   print("   Transfer complete.")






# In[31]:

#=========================<Main>================================================

def main():
    print("Starting style transfer program.")
    raw = getRawData()
    cData = preprocessData(raw[0])   # Content image.
    sData = preprocessData(raw[1])   # Style image.
    tData = preprocessData(raw[2])   # Transfer image.
    styleTransfer(cData, sData, tData)
    
    print("   Transfer complete.")
    print("Done. Goodbye.")



if __name__ == "__main__":
    main()


# In[ ]:




# In[ ]:




# In[ ]:



