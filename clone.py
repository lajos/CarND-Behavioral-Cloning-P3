import csv
import cv2
import numpy as np
from preprocess import preprocess

_training_data_root = 'training_data'

_cameras = { 0: 'center', 1: 'left', 2: 'right'}

class TrainData:
    def __init__(self):
        self.images = [[],[],[]]
        self.camera = []
        self.steer = []
        self.throttle = []
        self.brake = []
        self.speed = []

    def append(self, images, steer, throttle, brake, speed):
        for i in range(len(self.images)):
            self.images[i].append(images[i])
        self.steer.append(steer)
        self.throttle.append(throttle)
        self.brake.append(brake)
        self.speed.append(speed)

    def print(self, i):
        print('----------------------')
        print('train data :', i)
        print('  steer    :', self.steer[i])
        print('  throttle :', self.throttle[i])
        print('  brake    :', self.brake[i])
        print('  speed    :', self.speed[i])

    def get_train_data(self, camera):
        return (np.array(self.images[camera]),
            np.array(self.steer),
            # np.array(self.throttle),
            # np.array(self.brake),
            # np.array(self.speed)
            )

    def preprocess(self):
        for image_set_index in range(len(self.images)):
            for img_index in range(len(self.images[image_set_index])):
                self.images[image_set_index][img_index] = preprocess(self.images[image_set_index][img_index])



    def get_all_train_data(self):
        images = np.array(self.images[0]+self.images[1]+self.images[2]);
        steer = np.append(np.append(np.array(self.steer),np.zeros(len(self.steer))+0.15),np.zeros(len(self.steer))-0.15)
        return (images, steer)

    def get_train_data_straight_augmented(self):
        images = []
        steer = []
        center_steer_correction = 0.06
        for i in range(len(self.steer)):
            if abs(self.steer[i])<0.001:
                images.append(self.images[1][i])
                steer.append(center_steer_correction)
                images.append(self.images[2][i])
                steer.append(-center_steer_correction)
                pass
            else:
                images.append(self.images[0][i])
                steer.append(self.steer[i])
                images.append(np.fliplr(self.images[0][i]))
                #images.append(cv2.flip(self.images[0][i],1))
                steer.append(-self.steer[i])

        return(np.array(images), np.array(steer))

    def pickle(self):
        import pickle
        pickle.dump(self.images, open('images.p', 'wb'))
        pickle.dump(self.steer, open('steer.p', 'wb'))

    def unpickle(self):
        import pickle
        self.images = pickle.load(open('images.p', 'rb'))
        self.steer = pickle.load(open('steer.p', 'rb'))




def read_train_data_folder(train_data, folder):
    print('.read train data folder:',folder)
    i=0
    with open('{}/{}/driving_log.csv'.format(_training_data_root, folder)) as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        for line in reader:
            # if abs(float(line[3]))<0.001:
            #      continue
            images = []
            for image_name in line[:3]:
                image_name = image_name.split('\\')[-1]
                image_file = '{}/{}/IMG/{}'.format(_training_data_root, folder, image_name)
                image = cv2.imread(image_file)
                images.append(image)
            train_data.append(images, float(line[3]), float(line[4]), float(line[5]), (line[6]))
            i += 1
            if i==330:
                pass
#                break


def read_train_data(train_data, folders):
    for folder in folders:
        read_train_data_folder(train_data, folder)

train_data = TrainData()

_reload_data = False

if _reload_data:
    read_train_data(train_data,['1'])
    read_train_data(train_data,['2'])
    read_train_data(train_data,['1_rev'])
    read_train_data(train_data,['2_rev'])
    train_data.preprocess()
    train_data.pickle()
else:
    train_data.unpickle()


#X_train, y_train  = train_data.get_train_data(0)
#X_train, y_train  = train_data.get_all_train_data()
X_train, y_train  = train_data.get_train_data_straight_augmented()

print('X_train shape:',X_train.shape)
print('y_train shape:',y_train.shape)
print('X_train image shape:',X_train[0].shape)

#y_train = y_train * 1.2

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Convolution2D, Dropout, MaxPooling2D

model = Sequential()
model.add(Convolution2D(16, (3, 3), activation='relu', padding='same', input_shape=X_train.shape[1:]))
model.add(MaxPooling2D())
model.add(Convolution2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D())
model.add(Convolution2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D())
#model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=X_train.shape[1:]))
# model.add(Lambda(lambda x: x/255.0-0.5))
#model.add(Convolution2D(24, (5, 5), strides=(2,2), activation='relu'), padding='same')
#model.add(Convolution2D(36, (5, 5), strides=(2,2), activation='relu'))
#model.add(Convolution2D(48, (5, 5), strides=(2,2), activation='relu'))
#model.add(Convolution2D(64, (3, 3), strides=(1,1), activation='relu'))
#model.add(MaxPooling2D())
#model.add(Convolution2D(64, (3, 3), strides=(1,1), activation='relu'))
#model.add(MaxPooling2D())
# model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(1000))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

print(model.summary())
#from keras.utils import plot_model
#plot_model(model, to_file='model.png')

model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=10)

model.save('model.h5')


