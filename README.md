# **Traffic Sign Recognition** 

In this project, I implemented convolutional neural networks to classify traffic signs. I trained and validated a model so it can classify traffic sign images using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). After the model is trained, then I tried out my model on images of German traffic signs that I found on the web.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report




## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README



### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I calculated summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing the distribution of classes vursas training set

![alt text](https://i.ibb.co/YhPWQNq/index15.png)

### Design and Test a Model Architecture

#### 1. Processing the Data

As a first step, I decided to convert the images to grayscale because it's easier for classifier to learn 

Here is an example of a traffic sign image after grayscaling.

![alt text](https://i.ibb.co/KXpW3X9/15.png)

As a last step, I normalized the image data because calculating very low or high values numerically is a huge load , so it's easier for the optimizer
 



#### 2. Model architecture (Lenet).

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5	    |1x1 stride, valid padding, outputs 10x10x16  									|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Fully connected		| 4 Layers         									|
| Layer 1				| Flatten        									|
|layer 2			| output 120												|
|layer 3				|output 84										|
|layer 4				|output 43										| 

![alt text](https://i.ibb.co/mzQYD3y/lenet.png)


#### 3. Training the model.

To train the model, I used ..
1. Adam optimizer
2. Learning rate 0.001
3. 50 Epochs
4. Batch size is 80

Approach : First I converted the Images into grayscale , then I normalized them . the next step was designing the Architecture of Lenet , I modified the output to produce 43 Labels. Finally I start to train my model through calculating the cross entropy , then by optimizing my model using Adam which is similar to SGD with tuning Learning rate and number of epochs.

#### 4. Results.

My final model results were:
* training set accuracy is 99%
* validation set accuracy is 94%
* test set accuracy of 91%

Here I used Lenet architecture , here I tried to tune Learning rate , number of epochs.

I got a very good result with test accuracy 91% and I think it could be a traffic sign application.
 

### Test a Model on New Images

I tested the model with five images from the web , and it got 4 out of 5 correct
I tried to test the model with difficult shapes and low quality
you can find the images in ../test images
![alt text](https://i.ibb.co/hX7sy2z/11.png)
![alt text](https://i.ibb.co/Y84z5wj/22.png)
![alt text](https://i.ibb.co/zJc5Fkx/33.png)
![alt text](https://i.ibb.co/bHXnTwT/44.png)
![alt text](https://i.ibb.co/jyF1bJw/55.png)

#### 2. Result of prediction.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Right of way at next intersection     		| Right of way at next intersection  									| 
| speed Limit : 30     			| speed Limit : 30  										|
| turn left ahead 					| turn left ahead											|
|General caution	      		| General caution					 				|
| Road work			| no passing for vehicles    							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. 

#### 3. Images probabilities prediction



| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			|  Right of way at next intersection  									| 
| 1.00   				| speed Limit : 30  										|
| 1.00					| turn left ahead											|
| 1.00	      			| General caution					 				|
| .47				    | Road work     							|




