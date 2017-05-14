import csv
import cv2
import numpy as np
import random
from preprocess import preprocess
from pathlib import Path
from sklearn.utils import shuffle
import sys

_training_data_root = 'training_data'

_cameras = { 0: 'center', 1: 'left', 2: 'right'}

_batch_size = 256

def print_progress_bar (iteration, total, prefix = 'progress:', suffix = ' ', decimals = 1, length = 30, fill = '='):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '.' * (length - filledLength)
    print('\r%s [%s] %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    if iteration == total:
        print()

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
        return (np.array(self.images[camera]),np.array(self.steer))

    def preprocess(self):
        num_images=sum(len(x) for x in self.images)
        print('preprocess images:',num_images)
        i=0
        print_progress_bar(i, num_images, prefix = 'preprocess:')
        for image_set_index in range(len(self.images)):
            for img_index in range(len(self.images[image_set_index])):
                self.images[image_set_index][img_index] = preprocess(self.images[image_set_index][img_index])
                i+=1
                if i%100==0:
                    print_progress_bar(i, num_images, prefix = 'preprocess:')
        print_progress_bar(num_images, num_images, prefix = 'preprocess:')

    def get_all_train_data(self, side_steer=0.15):
        images = np.array(self.images[0]+self.images[1]+self.images[2]);
        steer = np.append(np.append(np.array(self.steer),
            np.zeros(len(self.steer))+side_steer),
            np.zeros(len(self.steer))-side_steer)
        print(images.shape)
        print(steer.shape)
        return (images, steer)

    def get_train_data_straight_augmented(self, correction=0.6, correction_range=0.01, side_correction=0.2):
        images = []
        steer = []
        center_steer_correction = correction
        print('.get train data steer augmented')
        print('    raw steer samples   :',len(self.steer))
        n_zero_steer=0
        n_with_steer=0
        for i in range(len(self.steer)):
            if abs(self.steer[i])<0.001:
                n_zero_steer+=1
                c = correction
                if correction_range>0:
                    c=random.uniform(correction-correction_range, correction+correction_range)
                images.append(self.images[1][i])
                steer.append(c)
                images.append(self.images[2][i])
                steer.append(-c)
            else:
                n_with_steer+=1
                images.append(self.images[0][i])
                steer.append(self.steer[i])
                images.append(np.fliplr(self.images[0][i]))
                steer.append(-self.steer[i])

                images.append(self.images[1][i])
                steer.append(min(1,self.steer[i]+side_correction))
                images.append(self.images[2][i])
                steer.append(max(-1,self.steer[i]-side_correction))
                images.append(np.fliplr(self.images[1][i]))
                steer.append(-(min(1,self.steer[i]+side_correction)))
                images.append(np.fliplr(self.images[2][i]))
                steer.append(-(max(-1,self.steer[i]-side_correction)))

        print('    no steer samples    :',n_zero_steer)
        print('    with steer samples  :',n_with_steer)
        print('    augmented samples   :',len(steer))
        return(np.array(images), np.array(steer))

    def sample_no_steer(self, i):
        correction = 0.065
        correction_range = 0.005
        c=random.uniform(correction-correction_range, correction+correction_range)
        r = random.randrange(2)
        if r==0:
            return(self.images[1][i], c)
        else:
            return(self.images[2][i], -c)

    def sample_with_steer(self, i):
        side_correction = 0.2
        r = random.randrange(6)
        if r==0:
            return(self.images[0][i], self.steer[i])
        elif r==1:
            return(np.fliplr(self.images[0][i]), -self.steer[i])
        elif r==2:
            return(self.images[1][i], min(1,self.steer[i]+side_correction))
        elif r==3:
            return(self.images[2][i], max(-1,self.steer[i]-side_correction))
        elif r==4:
            return(np.fliplr(self.images[1][i]), -(min(1,self.steer[i]+side_correction)))
        elif r==5:
            return(np.fliplr(self.images[2][i]), -(max(-1,self.steer[i]-side_correction)))

    def sample_generator(self):
        while 1:
            images = []
            steers = []
            for o in range(_batch_size):
                i = random.randrange(len(self.steer))
                s = self.steer[i]
                if abs(s)<0.001:
                    image, steer = self.sample_no_steer(i)
                else:
                    image, steer = self.sample_with_steer(i)
                images.append(image)
                steers.append(steer)
            yield(np.array(images), np.array(steers))


    def get_augmented_size(self):
        steer = np.array(self.steer)
        n_zero_steer = len(np.where(abs(steer)<0.001)[0])
        n_with_steer = len(steer) - n_zero_steer
        n_aug_samples = 2*n_zero_steer + 6*n_with_steer
        print('    samples             :',len(steer))
        print('    no steer samples    :',n_zero_steer)
        print('    with steer samples  :',n_with_steer)
        print('    augmented samples   :',n_aug_samples)
        return(n_aug_samples)

    def get_image_shape(self):
        return self.images[0][0].shape

    def pickle(self):
        import pickle
        pickle.dump(self.images, open('images.p', 'wb'))
        pickle.dump(self.steer, open('steer.p', 'wb'))

    def unpickle(self):
        import pickle
        self.images = pickle.load(open('images.p', 'rb'))
        self.steer = pickle.load(open('steer.p', 'rb'))


def get_csv_len(csv_filename):
    with open(csv_filename, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
        row_count = len(data)
    return(row_count)

def get_csv_filename(folder, force_marked=False):
    marked_file = Path('{}/{}/driving_log_marked.csv'.format(_training_data_root, folder))
    if force_marked or marked_file.is_file():
        return(str(marked_file))
    return '{}/{}/driving_log.csv'.format(_training_data_root, folder)

def read_train_data_folder(train_data, folder, min_speed=1):
    print('.read train data folder:',folder)
    csv_filename = get_csv_filename(folder)
    print('.drive log:',csv_filename)
    csv_len = get_csv_len(csv_filename)
    print('.csv length:', csv_len)
    i=0
    print_progress_bar(i, csv_len, prefix = 'read {}:'.format(folder))
    with open(csv_filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            try:
                speed = float(line[3])
            except:
                continue
            if float(line[6])<min_speed:
                continue
            marked = True
            if (len(line)>7):
                marked = not line[7]=='False'
                if not marked:
                    continue
            images = []
            for image_name in line[:3]:
                image_name = image_name.split('\\')[-1]
                image_file = '{}/{}/IMG/{}'.format(_training_data_root, folder, image_name)
                image = cv2.imread(image_file)
                images.append(image)
            train_data.append(images, float(line[3]), float(line[4]), float(line[5]), float(line[6]))
            i += 1
            if i%100==0:
                print_progress_bar(i, csv_len, prefix = 'read {}:'.format(folder))
    print_progress_bar(csv_len, csv_len, prefix = 'read {}:'.format(folder))

def read_train_data(train_data, folders):
    for folder in folders:
        read_train_data_folder(train_data, folder)

train_data = TrainData()

_reload_data = True

import time
start_time = time.time()

if _reload_data:
    read_train_data(train_data,['j1','j2','j3','j5','j6','j8','j9'])
    read_train_data(train_data,['k1','k2','k3'])
    read_train_data(train_data,['k4','k5','j4','j7'])  # shade turn
    read_train_data(train_data,['k6','k7','k8','k9'])  # downhill right
    read_train_data(train_data,['l1','l2', 'l3']) # downhill right
    read_train_data(train_data,['m1','m2', 'm3']) # shade left
    read_train_data(train_data,['n1','n2', 'n3', 'n4']) # shade right

    train_data.preprocess()

    train_data.pickle()
else:
    train_data.unpickle()


input_shape = train_data.get_image_shape()
print('input shape:',input_shape)

n_samples = train_data.get_augmented_size()

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Convolution2D, Dropout, MaxPooling2D
from keras import regularizers

_dropout=0.2

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=input_shape))
model.add(Convolution2D(8, (5, 5), activation='elu', padding='same'))
model.add(MaxPooling2D())
model.add(Dropout(_dropout))
model.add(Convolution2D(16, (3, 3), activation='elu', padding='same'))
model.add(MaxPooling2D())
model.add(Dropout(_dropout))
model.add(Convolution2D(32, (3, 3), activation='elu', padding='same'))
model.add(MaxPooling2D())
model.add(Dropout(_dropout))
model.add(Convolution2D(64, (3, 3), activation='elu', padding='same'))
model.add(MaxPooling2D())
model.add(Dropout(_dropout))
model.add(Flatten())
model.add(Dense(1000))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

print(model.summary())

for i in range(48):
    print('.epoch:',i+1)
    model.fit_generator(generator=train_data.sample_generator(),
        validation_data=train_data.sample_generator(),
        steps_per_epoch=int(n_samples/_batch_size*0.8),
        validation_steps=int(n_samples/_batch_size*0.2),
        epochs=1)
    if i>25:
       model.save('model.{:02d}.h5'.format(i))


model.save('model.h5')


