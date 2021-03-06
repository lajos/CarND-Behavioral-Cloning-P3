# **Behavioral Cloning Resubmission**

---

## Resubmission Notes

My previous submission was rejected for these reasons:
* normalized images outside of Keras
* did not use a generator

### Normalizing Outside of Keras

The reason why I normalized the images as part of preprocessing instead of using `Keras` was that due to a bug in `Keras`, `Lambda` layers cannot be saved on Windows.

When running this code:
```python
from keras.models import Sequential
from keras.layers import Lambda

model = Sequential()
model.add(Lambda(lambda x: (x/255.0)-0.5, input_shape = (2,2)))

model.save('model.h5')
```

Python throws the following error:

```
Traceback (most recent call last):
  File "c:\Projects\udacity\carnd\tmp\test.py", line 7, in <module>
    model.save('model.h5')
  File "C:\Dev\miniconda3\envs\carnd-term1\lib\site-packages\keras\engine\topology.py", line 2416, in save
    save_model(self, filepath, overwrite)
  File "C:\Dev\miniconda3\envs\carnd-term1\lib\site-packages\keras\models.py", line 101, in save_model
    'config': model.get_config()
  File "C:\Dev\miniconda3\envs\carnd-term1\lib\site-packages\keras\models.py", line 1176, in get_config
    'config': layer.get_config()})
  File "C:\Dev\miniconda3\envs\carnd-term1\lib\site-packages\keras\layers\core.py", line 668, in get_config
    function = func_dump(self.function)
  File "C:\Dev\miniconda3\envs\carnd-term1\lib\site-packages\keras\utils\generic_utils.py", line 177, in func_dump
    code = marshal.dumps(func.__code__).decode('raw_unicode_escape')
UnicodeDecodeError: 'rawunicodeescape' codec can't decode bytes in position 89-93: truncated \uXXXX
```

I ran into this bug early on while working on the project. I have fixed the bug locally, but I was afraid that my project files would not work during review, so I abandoned using `Lambda` layers. The fix:

Modify the following line in keras/utils/generic_utils.py:

```
code = marshal.dumps(func.__code__).decode('raw_unicode_escape')
```

To:

```
code = marshal.dumps(func.__code__).replace(b'\\',b'/').decode('raw_unicode_escape')
```

My submission includes `test.py` that reproduces the issue with the current version of `Keras`.

---

### Not Using a Generator

Project rubric: *"The code in model.py uses a Python generator, **if needed**, to generate data for training rather than storing the training data in memory."*

This sounds optional, not a requirement, to get around low memory issues. While working on this project it fit into memory comfortably, so I did not implement a generator.

![memory usage](writeup-assets/memory.png)

This screenshot shows the memory usage while training track2 with around 90k images. The spikes are `sklearn` reshuffling the data between epochs. Even during those spikes there is about 25% unused memory.

### Resubmission

I have removed the image normalization from preprocessing and added it to the model as a `Lambda` layer.

I implemented a sample generator for data augmentation. The generator also supports batching.

The resubmitted files are located in the `track2_resubmit` folder:
* **clone.py** - model training
* **drive.py** - steering prediction for simulator
* **preprocess.py** - image preprocessing used by clone.py and drive.py
* **test.py** - Keras library bug test while saving Lambda layers
* **model.h5** - Keras model data
* **track2_15.mp4** - screen recording from simulator @15mph
* **track2_20.mp4** - screen recording from simulator @20mph
* **track2_28.mp4** - screen recording from simulator @28mph

### Resubmission Results

---

Youtube video driving at 15mph, the driving samples were recorded at this speed:

[![track2 @15mph](./writeup-assets/track2_15.png)](https://youtu.be/I0-fxBHEar0 "track2 @15mph")

---

Youtube video driving at 20mph, this is 33% faster than the sample collection speed. The car oversteers, but is able to recover:

[![track2 @20mph](./writeup-assets/track2_20.png)](https://youtu.be/CLuOzqIrfIU "track2 @20mph")

---

Youtube video driving at 28mph, this is 87% faster than the sample collection speed. The car oversteers even more and at some point one of the rear wheels leave the road, but the car recovers. I recorded this clip only to see how far the model can be pushed:

[![track2 @25mph](./writeup-assets/track2_25.png)](https://youtu.be/r8nMST9UtVs "track2 @28mph")


---

# Original Writeup of the Project Follows

---

## Project

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering values from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

---

## Simulator

Udacity provided a simulator with 2 tracks.

![simulator](writeup-assets/simulator.jpg)

The simulator has two modes:
* **Training:** In training mode the car can be driven around the tracks by the user and the simulator records three images mounted in the center, left and right of the car and steering value, throttle, brake and speed values for each frame.
* **Autonomous:** In autonomous mode the simulator connects to an external program (drive.py), sends the image from the center mounted camera, the current speed and throttle position. The **drive.py** program uses the image to predict the steering value based on a trained [Keras](https://github.com/fchollet/keras) model and calculates the updated throttle position to maintain a predefined speed. The steering value and throttle position are then sent back to the simulator.

---

## Track 1

---

### Training Data Collection

After some failures and experimentation it became apparent that the quality of the training data is important to successfuly train the model.

I have found the XBOX controller to work best as an input method, but I don't play video games so it's been a challenge to drive the car around the track in the middle of the lane. I decided to "improve" the analog stick by attaching a 3d printed extension.

![xbox controller extension](writeup-assets/controller.jpg)

Data collected:
* 2 laps in normal direction, trying to stay in center of lane
* 2 laps in reverse direction
* recovery clips for exiting first turn, entering bridge, sharp turns

---

### Preprocessing

I created a separate preprocessing module (preprocess.py) to make sure that images are preprocessed the same way during training and prediction.

I tested several image spaces and found that converting the images to [HLS](https://en.wikipedia.org/wiki/HSL_and_HSV) provided the greatest improvement. This is one of the recorded images in RGB and HLS (displayed hue displayed as red, lightness as green, saturation as blue):

![rgb and hls](writeup-assets/rgb_hls.jpg)

I also cropped the bottom and top of the images to get rig of the front of the car and sky (lightness channel only):

![cropping](writeup-assets/crop.png)

Images are resized to 25% of their size (20,80,3) and normalized to zero mean and unit variance (values -0.5...0.5).

---

### Statistics
* samples colleted: 4,866
* samples driving straight: 1,068
* images with steering: 3,798
* augmented samples: **9,732**
* model trained for 16 epochs

---

### Data augmentation

When the steering wheel is centered (absolute value of steering is less than 0.001) I used the left and right camera images only with a low steering value to steer the car towards the center:

![left camera](writeup-assets/track1l.jpg)

*left camera, steering value 0.065*

![right camera](writeup-assets/track1r.jpg)

*right camera, steering value -0.065*

When the steering wheel is turned, only the center image is used. To make sure that the data set is balanced for left and right steering values, the image is also flipped horizontally and the steering value's sign negated:

![center camera](writeup-assets/track1a.jpg)

*center camera, steering value -0.5*

![center camera flipped](writeup-assets/track1b.jpg)

*flipped center camera, steering value 0.5*

---

### Initial model

My model was inspired by NVIDIA's model from their [End-to-End Deep Learning for Self-Driving Cars](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) article.

My graphics card does not have enough memory to run NVIDA's model and their model required a specific input image size so I have reduced the number of [convolutional layers](https://keras.io/layers/convolutional/#conv2d) and changed the padding to 'same' to allow more flexible experimentation with input image sizes.

I have added [max pooling](https://keras.io/layers/pooling/#maxpooling2d) and [dropout](https://keras.io/layers/core/#dropout) layers to prevent overfitting. Convolutional layers use RELU activation to allow non-linear training.

The model uses Adam optimizer with MSE (mean squared error) loss function.

My initial model architecture:

| layer                 |  parameters        | output       |
|:---------------------:|:------------------:|:------------:|
| Input                 | image              | 20, 80, 3    |
| Convolution RELU 1    | K=3, S=1, D=16     | 20, 80, 16   |
| MaxPooling            | K=2, S=2           | 10, 40, 16   |
| Convolution RELU 2    | K=3, S=1, D=32     | 10, 40, 32   |
| MaxPooling            | K=2, S=2           | 5, 20, 32    |
| Convolution RELU 3    | K=3, S=1, D=64     | 5, 20, 64    |
| MaxPooling            | K=2, S=2           | 2, 10, 64    |
| Dropout               | 0.2                |              |
| Flatten               |                    | 1280         |
| Dense                 |                    | 1000         |
| Dense                 |                    | 100          |
| Dense                 |                    | 50           |
| Dense                 |                    | 10           |
| Dense                 |                    | 1            |

**Total parameters:** 1,410,255

The model uses 20% of the samples for validation.

---

### Initial Result

The assets are located in the **track1_1** folder:
* **clone.py** - model training
* **drive.py** - steering prediction for simulator
* **preprocess.py** - image preprocessing used by clone.py and drive.py
* **model.h5** - keras model data
* **track1_1.mp4** - screen recording from simulator

Youtube video:

[![track1_1](./writeup-assets/track1_1_video.png)](https://youtu.be/5jPmGliQQh8 "track1_1")

---

### Improvements

I wanted to improve on my initial result because the car was not driving very smoothly. I collected additional driving data by driving the car around the track an additional time.

---

### Statistics
* samples colleted: 7,540
* samples driving straight: 1,957
* images with steering: 5,583
* augmented samples: **15,080**
* model trained for 17 epochs

I also added another convolutional layer and dropouts after each convolution. My revised model:

| layer                 |  parameters        | output       |
|:---------------------:|:------------------:|:------------:|
| Input                 | image              | 20, 80, 3    |
| Convolution RELU 1    | K=3, S=1, D=8      | 20, 80, 8    |
| MaxPooling            | K=2, S=2           | 10, 40, 8    |
| Dropout               | 0.1                |              |
| Convolution RELU 2    | K=3, S=1, D=16     | 10, 40, 16   |
| MaxPooling            | K=2, S=2           | 5, 20, 16    |
| Dropout               | 0.1                |              |
| Convolution RELU 3    | K=3, S=1, D=32     | 5, 20, 32    |
| MaxPooling            | K=2, S=2           | 2, 10, 32    |
| Dropout               | 0.1                |              |
| Convolution RELU 4    | K=3, S=1, D=64     | 2, 10, 64    |
| MaxPooling            | K=2, S=2           | 1, 5, 64     |
| Dropout               | 0.1                |              |
| Flatten               |                    | 1280         |
| Dense                 |                    | 1000         |
| Dense                 |                    | 100          |
| Dense                 |                    | 50           |
| Dense                 |                    | 10           |
| Dense                 |                    | 1            |

**Total parameters:** 451,199

The assets are located in the **track1_2** folder:
* **clone.py** - model training
* **drive.py** - steering prediction for simulator
* **preprocess.py** - image preprocessing used by clone.py and drive.py
* **model.h5** - keras model data
* **track1_2.mp4** - screen recording from simulator

Youtube video:

[![track1_1](./writeup-assets/track1_2_video.png)](https://youtu.be/2XNHo84qnU0 "track1_2")


---

## Track 2

---

### Challenges

Track 2 is more challenging because the turns are sharper, the track is longer, there track changes to deep downhill sections where the camera can't see the road and there are more dynamic lighting situations.

Due to my lacking experience with the game controller I collected data with the car running at 15 mph.

---

### Additional Preprocessing

#### Shadows

There are lighting scenarios where part or all of the track is in shadow. I have used CLAHE ([Contrast Limited Adaptive Histogram Equalization](https://en.wikipedia.org/wiki/Adaptive_histogram_equalization)) on the lightness channel to minimize the affect of shadows.

![lightness](writeup-assets/L.png)

*lighness channel*

![lightness with clahe](writeup-assets/L_CLAHE.png)

*lighness channel with CLAHE applied*

#### Roadside Distractions

There are a lot of distracting features on the sides, for example trees. To minimize this I have use a projection transformation to "un-tilt" the images:

![untilt](writeup-assets/untilt.png)

#### Tagging Good Samples

NVIDIA's [End-to-End Deep Learning for Self-Driving Cars](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) article mentions that they only use center lane driving samples. To make that sure I am training the model on good data, I wrote a small python program to play through the samples and tag samples to use or ignore.

The **tag_data.py** script loads a folder of samples recorded by the simulator and displays the frames, frame numbers and steering values in a window. The frames can be played forward and back (**left** and **right arrow** keys) and tagged for use (**up arrow**) or ignore (**down arrow**).

The bottom bar's color indicates if a frame should be used: red for good data and gray for the sample to be ignored.

The **return** key saves a driving_log_marked.csv file that has an extra boolean attribute to indicate if the sample should be used.

![data tagger](writeup-assets/data_tagger.jpg)

*the left image (frame 0, steering value 0) is ignored, the right image (frame 96, steering value 0.8) is used*

#### Side Cameras

The simulator records images from 3 cameras: center, left, right.

I used the images from the side cameras with a steering offset. The center cameras's steering value for the left camera's are increased by 0.2, the right cameras reduced by 0.2.

![center](writeup-assets/center.jpg)

*center camera, steering value 0.3*

![left](writeup-assets/left.jpg)

*left camera, steering value 0.3+0.2=0.5*

![right](writeup-assets/right.jpg)

*right camera, steering value 0.3-0.2=0.1*

#### Samples While Autonomous

There were some tricky turns where the car got into trouble no matter how many clips I recorder in training mode.

![tricky](writeup-assets/tricky.jpg)

*in this turn the camera can't see the road because it turns and descends*

For these situations I recorded images in autonomous mode and assigned a steering value that seemed appropriate.

#### Data Augmentation

The following images were added to the training set:

If the steering value is close to zero:
* left camera, steering value of 0.065
* right camera, steering value -0.065

If the magnitude of steering value is greater than zero:
* center camera, steering value as recorded
* center camera flipped horizontally, steering value negated
* left camera, steering value + 0.2
* left camera flipped, -(steering value + 0.2)
* right camera, steering value - 0.2
* right camera flipped, -(steering value - 0.2)

---

### Data Collection

* 4 laps around the track
* 20 recovery clips

---

### Statistics
* samples colleted: 16,477
* samples driving straight: 2,772
* images with steering: 13,705
* augmented samples: **87,774**
* model trained for 34 epochs

---

### Model Architecture

I have changed the activation for convolutional layers to [ELU](http://image-net.org/challenges/posters/JKU_EN_RGB_Schwarz_poster.pdf), doubled the depth of convolutional layers and doubled the input image size to 32x160x3.

The data is reshuffled between epochs to avoid validating to the same set of images.

My final model architecture:

| layer                 |  parameters        | output       |
|:---------------------:|:------------------:|:------------:|
| Input                 | image              | 32, 160, 3   |
| Convolution ELU 1     | K=3, S=1, D=8      | 32, 160, 8   |
| MaxPooling            | K=2, S=2           | 16, 80, 8    |
| Dropout               | 0.2                |              |
| Convolution ELU 2     | K=3, S=1, D=16     | 16, 80, 16   |
| MaxPooling            | K=2, S=2           | 8, 40, 16    |
| Dropout               | 0.2                |              |
| Convolution ELU 3     | K=3, S=1, D=32     | 8, 40, 32    |
| MaxPooling            | K=2, S=2           | 4, 20, 32    |
| Dropout               | 0.2                |              |
| Convolution ELU 4     | K=3, S=1, D=64     | 4, 20, 64    |
| MaxPooling            | K=2, S=2           | 2, 10, 64    |
| Dropout               | 0.2                |              |
| Flatten               |                    | 1280         |
| Dense                 |                    | 1000         |
| Dense                 |                    | 100          |
| Dense                 |                    | 50           |
| Dense                 |                    | 10           |
| Dense                 |                    | 1            |

**Total parameters:** 1,411,583

---

### Results

The assets are located in the **track2** folder:
* **clone.py** - model training
* **drive.py** - steering prediction for simulator
* **preprocess.py** - image preprocessing used by clone.py and drive.py
* **tag_data.py** - script for tagging samples
* **model.h5** - keras model data
* **track2_15.mp4** - screen recording from simulator at 15mph
* **track2_20.mp4** - screen recording from simulator at 20mph
* **track2_25.mp4** - screen recording from simulator at 25mph

---

Youtube video driving at 15mph, the driving samples were recorded at this speed:

[![track2 @15mph](./writeup-assets/track2_15.png)](https://youtu.be/ztb1CK3CLtU "track2 @15mph")

---

Youtube video driving at 20mph, this is 33% faster than the sample collection speed. The car oversteers, but is able to recover:

[![track2 @20mph](./writeup-assets/track2_20.png)](https://youtu.be/YaDN41ilTG4 "track2 @20mph")

---

Youtube video driving at 20mph, this is 66% faster than the sample collection speed. The car oversteers even more, but it stays on the road. This is the highest speed this model can drive:

[![track2 @25mph](./writeup-assets/track2_25.png)](https://youtu.be/0NYE1wGkuz0 "track2 @25mph")

---

### Github Repo of Project

[lajos/CarND-Behavioral-Cloning-P3](https://github.com/lajos/CarND-Behavioral-Cloning-P3)

---
