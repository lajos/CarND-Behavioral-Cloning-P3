# **Behavioral Cloning**

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

---

## Model Architecture

My model was inspired by NVIDIA's model from their [End-to-End Deep Learning for Self-Driving Cars](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) article.

My graphics card does not have enough memory to run NVIDA's model and their model required a specific input image size so I have reduced the number of [convolutional layers](https://keras.io/layers/convolutional/#conv2d) and changed the padding to 'same' to allow more flexible experimentation with input image sizes.

I have added [max pooling](https://keras.io/layers/pooling/#maxpooling2d) and [dropout](https://keras.io/layers/core/#dropout) layers to prevent overfitting.

My final model architecture:

| layer                 |  parameters        | output     |
|:---------------------:|:------------------:|:----------:|
| Input                 | image              | 160x32x3   |
| Convolution 1         | K=3, S=1, D=64     | 32x32x64   |
| RELU                  |                    |            |
| MaxPool 1             | K=2, S=2           | 16x16x64   |
| Dropout               | 0.6                |            |
| Convolution 2         | K=3, S=1, D=128    | 16x16x128  |
| RELU                  |                    |            |
| MaxPool 2             | K=2, S=2           | 8x8x128    |
| Dropout               | 0.5                |            |
| Convolution 3         | K=3, S=1, D=256    | 8x8x256    |
| RELU                  |                    |            |
| MaxPool 3             | K=2, S=2           | 4x4x256    |
| Dropout               | 0.6                |            |
| Convolution 4         | K=3, S=1, D=512    | 4x4x512    |
| RELU                  |                    |            |
| MaxPool 4             | K=2, S=2           | 2x2x512    |
| Dropout               | 0.6                |            |
| Flatten               | concat MaxPool 1-4 | 30720      |
| Fully Connected 1     | L2 regularization  | 4096       |
| RELU                  |                    |            |
| Dropout               | 0.6                |            |
| Fully Connected 2     | L2 regularization  | 1024       |
| RELU                  |                    |            |
| Dropout               | 0.7                |            |
| Fully Connected 3     | L2 regularization  | 43         |
| Sigmoid               |                    |            |

All convolution and max pooling layers use 'SAME' padding and K and S values are same on x and y axis.

---

### Files Submitted

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md or writeup_report.pdf summarizing the results

### Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 18-24)

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18).

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21).

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ...

For details about how I created the training data, see the next section.

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

To combat the overfitting, I modified the model so that ...

Then I ...

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
