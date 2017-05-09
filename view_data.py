#!/usr/bin/python

import tkinter as tk
from PIL import Image, ImageTk
from tkinter.ttk import Frame, Button, Style
import time
import csv
from pathlib import Path
import cv2

_training_data_root = 'training_data'

def print_progress_bar (iteration, total, prefix = 'progress:', suffix = ' ', decimals = 1, length = 30, fill = '='):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '.' * (length - filledLength)
    print('\r%s [%s] %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    if iteration == total:
        print()

class Display():
    def __init__(self, train_data):
        self.root = tk.Tk()
        self.root.title('training data')
        self.hud_text = tk.StringVar()
        self.train_data = train_data

        self.image = train_data.get_center(0)

        self.current_frame = 0

        w = self.image.width()
        h = self.image.height()

        self.root.geometry("%dx%d+%d+%d" % (w, h+20, 0, 0))

        self.image_label = tk.Label(self.root, image=self.image, borderwidth=0)

        self.root.bind('<Right>', self.forward)
        self.root.bind('<Left>', self.back)
        self.root.bind('<Up>', self.mark)
        self.root.bind('<Down>', self.unmark)
        self.root.bind('<Return>', self.save)

        self.hud_label=tk.Label(self.root, textvariable=self.hud_text, bg='black', fg='white',borderwidth=0, height=20)

        self.image_label.pack(side=tk.TOP, fill=tk.BOTH, expand=tk.YES)
        self.hud_label.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=tk.YES)

        self.image_label.focus_set()

        self.update_hud()

        self.root.mainloop()

    def update_hud(self):
        steer = self.train_data.get_steer(self.current_frame)
        text = '{}   {:.1f}'.format(self.current_frame, steer)
        self.hud_text.set(text)
        marked = self.train_data.get_marked(self.current_frame)
        if marked:
            self.hud_label.configure(bg='red')
        else:
            self.hud_label.configure(bg='gray')

    def set_frame(self, frame_number):
        if frame_number >= self.train_data.get_len():
            self.current_frame = 0
        elif frame_number<0:
            self.current_frame = self.train_data.get_len()-1
        else:
            self.current_frame = frame_number
        self.image = self.train_data.get_center(self.current_frame)
        self.image_label.configure(image=self.image)
        self.update_hud()

    def forward(self, event):
        self.set_frame(self.current_frame+1)

    def back(self, event):
        self.set_frame(self.current_frame-1)

    def mark(self, event):
        self.train_data.set_marked(self.current_frame, True)
        self.set_frame(self.current_frame+1)

    def unmark(self, event):
        self.train_data.set_marked(self.current_frame, False)
        self.set_frame(self.current_frame+1)

    def save(self, event):
        save_train_data(train_data)

class TrainData:
    def __init__(self):
        self.folder = None
        self.center = []
        self.left = []
        self.right = []
        self.images = [[],[],[]]
        self.camera = []
        self.steer = []
        self.throttle = []
        self.brake = []
        self.speed = []
        self.marked=[]

    def append(self, center, left, right, images, steer, throttle, brake, speed, marked):
        for i in range(len(self.images)):
            self.images[i].append(images[i])
        self.center.append(center)
        self.left.append(left)
        self.right.append(right)
        self.steer.append(steer)
        self.throttle.append(throttle)
        self.brake.append(brake)
        self.speed.append(speed)
        self.marked.append(marked)

    def get_center(self, index):
        image = self.images[0][index]
        pimage = Image.fromarray(image)
        return(ImageTk.PhotoImage(image=pimage))

    def get_steer(self, index):
        return(self.steer[index])

    def get_marked(self, index):
        return(self.marked[index])

    def set_marked(self, index, value):
        self.marked[index] = value

    def get_len(self):
        return(len(self.images[0]))

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

def save_train_data(train_data):
    folder = train_data.folder
    print('.save train data to folder:', folder)
    csv_filename = get_csv_filename(folder,force_marked=True)
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for i in range(train_data.get_len()):
            data = []
            #center, left, right, steer, throttle, brake, speed, marked
            data.append(train_data.center[i])
            data.append(train_data.left[i])
            data.append(train_data.right[i])
            data.append(train_data.steer[i])
            data.append(train_data.brake[i])
            data.append(train_data.throttle[i])
            data.append(train_data.speed[i])
            data.append(train_data.marked[i])
            writer.writerow(data)

def read_train_data(train_data, folder, min_speed=15):
    print('.read train data folder:',folder)
    train_data.folder = folder
    csv_filename = get_csv_filename(folder)
    print(csv_filename)
    csv_len = get_csv_len(csv_filename)
    i=0
    print_progress_bar(i, csv_len, prefix = 'read {}:'.format(folder))
    with open(csv_filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            try:
                speed = float(line[3])
            except:
                continue
            images = []
            for image_name in line[:3]:
                image_name = image_name.split('\\')[-1]
                image_file = '{}/{}/IMG/{}'.format(_training_data_root, folder, image_name)
                image = cv2.imread(image_file)
                b,g,r = cv2.split(image)
                image = cv2.merge((r,g,b))
                #image = Image.open(image_file)
                images.append(image)
            marked = True
            if (len(line)>7):
                marked = not line[7]=='False'
            train_data.append(line[0], line[1], line[2], images, float(line[3]), float(line[4]), float(line[5]), (line[6]), marked)
            i += 1
            if i%100==0:
                print_progress_bar(i, csv_len, prefix = 'read {}:'.format(folder))
            # if i==101:
            #     break
    print_progress_bar(csv_len, csv_len, prefix = 'read {}:'.format(folder))

if __name__ == '__main__':
    train_data = TrainData()
    read_train_data(train_data,'j7')
    app = Display(train_data)
