# Handwritten Digits Recognition using Keras CNN  
### - Introduction:  
Constructing a **Machine Learning model** or a **Deep Learning model** or an **Artificial Neural Network**, that recognizes the digits from images, is considered to be the "Hello World!" of any field from the mentioned before. Therefore, in order to improve your skills and experinces, you should start the thousand miles journey from this.  

### - Dataset:  
Here, I used the MNIST HANDWRITTEN DIGITS dataset. The MNIST database (Modified National Institute of Standards and Technology database) is a large database of handwritten digits that is commonly used for training various image processing systems. The database is also widely used for training and testing in the field of machine learning. It was created by "re-mixing" the samples from NIST's original datasets.
*- Wikipedia*  
The next figure shows a peak look into MNIST dataset:  
  
![Image of Yaktocat](https://cdn-images-1.medium.com/max/800/0*At0wJRULTXvyA3EK.png)  
  
### - Specifications:  
The dataset contains more than 60K images for hand-written digits. Each image is in shape of ```(784, 1)```, in this case, we must reshape the image into ```(28, 28)``` shape to treat it as an image. The dataset is already splitted into **Train, Test, Validation**.  
  
### - Work:  
  
### -- Convolutional Neural Network architecture:  
  
```python
print(new_model.summary())
```  
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_5 (Conv2D)            (None, 28, 28, 16)        416       
_________________________________________________________________
activation_9 (Activation)    (None, 28, 28, 16)        0         
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 5, 5, 16)          0         
_________________________________________________________________
conv2d_6 (Conv2D)            (None, 1, 1, 32)          12832     
_________________________________________________________________
activation_10 (Activation)   (None, 1, 1, 32)          0         
_________________________________________________________________
dropout_5 (Dropout)          (None, 1, 1, 32)          0         
_________________________________________________________________
flatten_3 (Flatten)          (None, 32)                0         
_________________________________________________________________
dense_5 (Dense)              (None, 512)               16896     
_________________________________________________________________
activation_11 (Activation)   (None, 512)               0         
_________________________________________________________________
dropout_6 (Dropout)          (None, 512)               0         
_________________________________________________________________
dense_6 (Dense)              (None, 10)                5130      
_________________________________________________________________
activation_12 (Activation)   (None, 10)                0         
=================================================================
```  
  
### -- Evaluation over validation set:  
  
```python
print('='*50)
print('>> Evaluate on validation data: (WHICH NEVER SEEN)')
results = new_model.evaluate(X_valid, Y_valid, batch_size=128)
print('Accuracy: %0.4f%%' % (results[1]*100))
print('Loss: %0.4f%%' % (results[0]*100))
print('='*50)
```
```
==================================================
>> Evaluate on validation data: (WHICH NEVER SEEN)
5000/5000 [==============================] - 1s 295us/step
Accuracy: 98.6200%
Loss: 5.1526%
==================================================
```
  
### -- Here are some predictions of images that *NEVER SEEN BY NEURAL NETWORK*:  
  
![Output](https://i.ibb.co/Lnnd7LJ/download.png)
