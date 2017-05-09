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

    def get_train_data_straight_augmented(self, correction=0.6, correction_range=0.01):
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

        print('    no steer samples    :',n_zero_steer)
        print('    with steer samples  :',n_with_steer)
        print('    augmented samples   :',len(steer))
        return(np.array(images), np.array(steer))

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

if _reload_data:
    read_train_data(train_data,['j1','j2','j3','j4','j5','j7'])
#    read_train_data(train_data,['j3'])

    train_data.preprocess()
    train_data.pickle()
else:
    train_data.unpickle()


#X_train, y_train  = train_data.get_all_train_data(side_steer=0.2)
X_train, y_train  = train_data.get_train_data_straight_augmented(correction=0.065, correction_range=0.005)


print('X_train shape:',X_train.shape)
print('y_train shape:',y_train.shape)
print('X_train image shape:',X_train[0].shape)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Convolution2D, Dropout, MaxPooling2D

_dropout=0.2

model = Sequential()
model.add(Convolution2D(8, (5, 5), activation='elu', padding='same', input_shape=X_train.shape[1:]))
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

# nvidia
# model.add(Convolution2D(3, (5, 5), strides=(2,2), activation='elu',input_shape=X_train.shape[1:]))
# model.add(Convolution2D(24, (5, 5), strides=(2,2), activation='elu'))
# model.add(Convolution2D(36, (5, 5), strides=(2,2), activation='elu'))
# model.add(Convolution2D(48, (3, 3), activation='elu'))
# model.add(Convolution2D(64, (3, 3), activation='elu'))

model.add(Flatten())
model.add(Dense(1000))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

print(model.summary())

#sys.exit(0)

for i in range(64):
    print('.epoch:',i+1)
    X_train, y_train = shuffle(X_train, y_train)
    model.fit(X_train, y_train, validation_split=0.2, shuffle=False, epochs=1)
    if i%2==1:
       model.save('model.{:02d}.h5'.format(i))

model.save('model.h5')


