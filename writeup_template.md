# **Traffic Sign Recognition** 

Project: Carlos Fernandez

Udacity **Self Driving Cars**

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeupimages/signals.png "Traffic Signs"
[image2]: ./writeupimages/barchart_1.png "Barchart labels"
[image3]: ./writeupimages/color_gray_comparison.png "Color gray comparison"
[image4]: ./writeupimages/piechart.png  "Training, test and validation splits"
[image5]: ./writeupimages/loss_acuracy_training_model.png "Loss and Accuracy when training"
[image6]: ./writeupimages/new_signs.png "New Signs"
[image7]: ./writeupimages/new_signs_gray.png "New Signs grayed"
[image8]: ./signs/12_2.jpg "Traffic Sign 1"
[image9]: ./signs/3_1.jpg "Traffic Sign 2"
[image10]: ./signs/d.jpg "Traffic Sign 3"
[image11]: ./signs/g.jpg "Traffic Sign 4"
[image12]: ./signs/s.jpg "Traffic Sign 5"

## Rubric Points

### Writeup / README

#### 1. Writeup and project code

This document is the writeup of the Project "Traffic Sign Recognition".  It explains the process for developing a tensorflow based model to correctly classify german traffic signs.

Link to my [project code](https://github.com/crandomize/CarND-Traffic-Sign-Classifier-Project)

### Data Set Summary & Exploration

#### 1. Summary of the data set.

The code for this step is contained in the 3rd cell in the IPython notebook 

Numpy and matplotlib were used to generate a summary detailing number of images in the train and test data sets for each of the available labels.
The proportion of different traffic signs across the two datasets is constant, but there's a clear unbalance between the number of samples per traffic sign in each dataset.

* The size of training set is 39209
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

#### 2. Exploratory visualization of the dataset.

The code for this step is contained in 5th and 6th cells of the IPython notebook.  

First plot shows the number of samples per each of the available labels in the datasets. 
Training data set in blue and testing data set in red

![Barchart showing number of samples per class][image2]

Second plot shows an example per each of the labels.  This is done in training set so we can quicly lookup possible signs and its labels.

![Sample of each class][image1]

### Design and Test a Model Architecture

#### 1. Preprocessing

The code for this step is contained in the 7th and 8th code cells of the IPython notebook.
2 steps are implemented in the preprocessing:
- Gray scaling
- Normalization

First we do a **grayscaling** of the image.  This will allow to speedup the process of training as the number of initial layers will be reduce to 1.
We tested both color and gray scaled images and found that graying the images improved final accuracy which may indicate that model is more sensible to forms and images rather than different colors found on the signs (red, blue, black and white colors)

Below an image after the grayscaling process.

Here is an example of a traffic sign image before and after grayscaling.

![Color and gray image][image3]

After the gray scaling we do a **normalization** of the image.  We have tried different normalization values and finally decided to normalize all values between -1 and 1.


#### 2. Datasets split and data augmentation

##### 2.1 Data sets
The code for splitting the data into training and validation sets is contained in 9th and 10th code cell of the IPython notebook.  

I have splitted the testing set into 2 sets, the final training and the validation sets.  I have prefer not to use the training set to increase number of training samples in the learning step.
The split was as follows:

- Get a list of indexes from the test set (sequence)
- Shuffle the list of indexes
- Get final images splitting the shuffled index list in 2, one for validation and one for testing.
- Use same indexes splitted for the labels.

After the splitting we endup with 3 different sets:

- Training set: Use for training the initial model
- Validation set: For getting accuracies within the hyperparameter setup
- Testing set: For final accuracy estimation (once hyperparameters are frozen)

The final final partitions sizes (before data augmentation) are described in following piechart:

![Training, test and validation sets sizes][image4]

##### 2.1 Data Augmentation

Data Augmentation is done in cell 11 from the IPython Notebook.
As DN need to be feed with lots of data to improve accuracy I have added new data based on the existing training data. These new data is composed of images rotated in different angles. The final training set was then composed by:

- original test set
- images from test set rotated: -30, -20, -10, 10, 20, 30 degrees.

Total number of final data for training is: **274463**

The improvement in accuracy was very noticeable.


#### 3. Model architecture

The code for my final model is located in the 13th cell of the IPython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Gray image   							| 
| Convolution    	    | Input = 32x32x1 Output = 28x28x8 (Filter: 5x5x8) 	|
| Activation			| RELU											|
| Max pooling	      	| 2x2 stride,  Output: 14x14x8  				|
| Convolution    	    | Input = 14x14x8 Output = 10x10x16 (Filter: 5x5x16)	|
| Activation			| RELU											|
| Convolution    	    | Input = 10x10x16 Output = 7x7x32 (Filter: 4x4x32)	|
| Activation			| RELU											|
| Fully Connected       | Input = 1568 (4x4x32)  Output = 120 |
| Activation			| RELU											|
| Drop OUT			| keep 50%											|
| Fully Connected       | Input = 120   Output = 84 |
| Activation			| RELU											|
| Fully Connected       | Input = 84  Output = 43 |
| Output |logits|

The model is a variation of the LeNET architecture.  In order to decrease overfitting we have added a DropOut layer.
Also added a fully connected layer and deleted one of the Max Pooling layers (from the LeNET)

#### 4. Training the model and hyperparameters.

The code for training the model is located in cell 27th of the Ipython notebook. 

***Optimizer:***
For training the model I used a AdamOptimizer.   After comparing both the AdamOptimizer and GradientDescentOptimizer I found that the first one gives much better results.  See below figures of averaces accuracies.

- AdamOptimizer : 0.949  
- GradientDescentOptimizer: 0.88  

For evaluating the loss I used the average cross-entropy between the target and the softmax activation function applied to the prediction (tf.nn.softmax_cross_entropy_with_logits)

***Batch size:***
Although theory suggest that the larger the batch size (up to what GPU mem supports) the better results, I didn't found clear improvements when using 256 or 512 samples per batch.  So I keep the batch size to 128 images

Batch Size: 128

***Epochs***
Number of Epochs is reduced from original number 10 to 5.  Valiation accuracy was not improved after that and increasing the number of Epochs could make the model to overfit.

Epochs: 5

***Learning rate***
Different values were tested, best value seem to be 0.001

Learning rate: 0.001

***DropOut***
In the first fully connected layer I added a dropout of 50%

pkeep: 0.50

#### 5. Description in the approach taken for finding a solution.

***Process:***
I used training set for feeding the model learning process and calculate the accuracy on each Epoch. Accuracy in each Epoch is calculated first with the training set and then again with the validation set.  This information is valuable to detect when system overfits.
I kept iterating changing model layers as well as hyperparameters trying to improve valiation accuracy and while not overfiting.
Once results seem correct, then I calculate the accuracy on the test set.

See below the final Loss and Accuracy plot on final model when in training phase both for Train and Validation sets.

![Loss and Accuracy][image5]

***Model:***
I started with LeNet model and kept changing adding/removing layers.
As number of total output classes was larger than original LeNet, I tried to add some additional fully connected layers at the end without much success on accuracy improvements. I finally added one additional convolutional layer, while moving out one pooling layer.  Adding/removing layers and testing its accuracy on the validation set was the basic mode of operation.

I didn't appreciate big improvements, at least compared with data preprocessing efforts (moving to gray images) and specially data aumentation, which increased substatially the final accuracy of the model.

Adding drop out to the first fully connected layer  moved from around 0.92 to 0.95 accuracy improvement.

***Future work:***
As future work on this I plan to do some U type of model as suggested in the "Traffic Sign Recognition with Multi-Scale Convolutional Networks" from Sermanet/LeCunn.


### Test a Model on New Images

#### 1. New Images

Here are five German traffic signs that I found on the web:
 ![New images][image6]

We initially expect to be easy for the model to classify these signs.  We have cropped the images besides the 4th sign.

Again, as done in original images we gray the images and normalize them before feeding them into the model.

 ![Grayed images][image7]


#### 2. Model's predictions on these new traffic signs and compare the results to predicting on the test set. 

The code for making predictions on my final model is located in the 21st cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Right of way      	| Right of way    							| 
| Speed limit (60km/h)	| Speed limit (60km/h)						|
| Pedestrians			| Pedestrians								|
| No vehicles      		| **Slippery road** (ERROR) 			 				|
| Road work			    | Road work      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This is lower than the 95% accuracy on the test set.
Given that new images sources are totally different from those from the training sets this decrease on accuracy could be normal.
Additionally model generalization is not good enough for new images (not cropped perfectly, etc) and some overfitting on "how" the training images were preassembled (i.e. crop) is one of the reasons.

In any case the reduced number of this new set is not enough to have any statistical significance and a higher number of images should be used.

#### 3. Model prediction probabilities.

The code for making predictions on my final model is located in the 22nd cell of the Ipython notebook.

##### 1st image
The model correcly predicts this image with 100% certainty. 

![Grayed images][image8]

|Rank|Prediction|Probability|
|:----:|:----:|:---------------------------------------------:| 
|1| Right-of-way at the next intersection (11)| 100.0% |  
|2| Beware of ice/snow (30)| 0.0%  | 
|3| Turn left ahead (34)| 0.0%|

##### 2nd image

The model correcly predicts this image, but certainty is much lower, in fact closely followed by Turn left ahead sign with just 5% of difference in the certainty level.

![Grayed images][image9]

|Rank|Prediction|Probability|
|:----:|:----:|:---------------------------------------------:| 
|1| Speed limit (60km/h) (3)| 52.6%   |
|2| Turn left ahead (34)| 47.2%   |
|3| Keep right (38)| 0.1%|

##### 3rd image
The model correcly predicts this image with 100% certainty. 

![Grayed images][image10]

|Rank|Prediction|Probability|
|:----:|:----:|:---------------------------------------------:| 
|1| Pedestrians (27) |100.0%   |
|2| Right-of-way at the next intersection (11) |0.0%   |
|3| Dangerous curve to the left (19)| 0.0%|

##### 4th image

Model is **not able to correctly predic** this sign and correct answer is not even in the top 3 predictions.
This may be due to not be a cropped image.

![Grayed images][image11]

|Rank|Prediction|Probability|
|:----:|:----:|:---------------------------------------------:| 
|1| End of all speed and passing limits (32) |95.6%   |
|2| End of no passing (41) |4.1%   |
|3| No passing (9)| 0.1%|

##### 5th image

The model correcly predicts this image with almost a 100% certainty. 

![Grayed images][image12]

|Rank|Prediction|Probability|
|:----:|:----:|:---------------------------------------------:| 
|1| Road work (25) |99.9%|   
|2| General caution (18)| 0.1%   |
|3| Wild animals crossing (31) |0.0%|

