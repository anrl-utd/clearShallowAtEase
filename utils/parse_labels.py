import glob, os, shutil
import cv2
import random
from PIL import Image
from group import *

'''
We parse labels, and save their described bounding boxes as individual images. Has 
to be re-run individually for every class, but eventually wil probably be put in a func
so we can iterate through a list of classes. We only have two currently so I was lazy.
'''

class_label = 'car_'
labels_dir = '/home/sid/datasets/mvmc_p/'
images_dir = '/home/sid/datasets/multiview/'
save_dir = '/home/sid/datasets/mvmc_p/'

# clean dirs
def clean(dir_):
    for the_file in os.listdir(dir_):
        file_path = os.path.join(dir_ + the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)

clean(save_dir + 'train_dir/' + class_label + 'images/')
clean(save_dir + 'test_dir/' + class_label + 'images/')

frames_read = [0 for number in range(244)]

for filename in os.listdir(os.path.join(labels_dir, class_label + 'bounds/')):
    parsed_name = filename
    parsed_name = parsed_name.split('_')
    # data is of the form ['det','frameXX','camX']
    # we need to fetch img from frameXX of camX
    # frame is stored in form 00000..num.jpg
    frame = parsed_name[1].replace('frame','')
    zeros = 8 - len(frame)

    # reserving just frame number for future use
    frame_num = int(frame)

    # add proper formating to read from images in save_dir
    frame = ('0'*zeros) + frame + '.jpg'
    camera = parsed_name[2].replace('cam','c').replace('.txt', '')  # since folder names are c0-c5

    g = ''
    f = open(os.path.join(labels_dir, class_label + 'bounds/', filename), 'r')
    for line in f:
        g += line
    coords = g.strip().split()

    sz = len(coords)
    boxes = []
    i = 0
    # the number of bounding boxes is len(coords) / 4 because each bounding box is 4 args
    while i < sz:
        i_bound = []
        for index in range(4):
            i_bound.append(int(float(coords[i+index])))   
        i += 4
        boxes.append(i_bound)

    # so we don't modify actual save directory, we clone it and mess with that
    saved_dir = save_dir
    
    change_dir = 1
    # keep a global list of frames we have read, we need to keep same frames in the same directory
    if frames_read[frame_num] == 0:
        # roughly 80-20 train/test split
        r = random.uniform(0, 1)
        if r > 0.8:
            saved_dir += 'test_dir/' + class_label + 'images/'
        else:
            saved_dir += 'train_dir/' + class_label + 'images/'
     
        frames_read[frame_num] = saved_dir
    else:
        # if the frame is already in our global frame list, since we need to keep same frames
        saved_dir = frames_read[frame_num]
 
    # boxes now (theoretically) contains a list of bounding boxes
    for index in range(len(boxes)):
        y = boxes[index][1]
        x = boxes[index][0]
        w = boxes[index][2] - x
        h = boxes[index][3] - y
       
        orig_frame_dir = images_dir + camera + '/' + str(frame)
        orig_frame = cv2.imread(orig_frame_dir)
        cropped_frame = orig_frame[y:y+h, x:x+w]
        save_name = saved_dir + camera + '_' + str(index) + '_' + (frame)
        print(save_name)
        width, height, channels = cropped_frame.shape
        if height == 0 or width == 0:
            print('here')
            continue
        print("size: " + str(width) + ", " + str(height) + ", " + str(channels))
        cropped_frame = cv2.resize(cropped_frame, (32,32), interpolation=cv2.INTER_CUBIC)
        width, height, channels = cropped_frame.shape
        print("size: " + str(width) + ", " + str(height) + ", " + str(channels))
        cv2.imwrite(save_name, cropped_frame)

# calling function from group.py
group_frames(save_dir + 'train_dir/' + class_label + 'images/')
group_frames(save_dir + 'test_dir/' + class_label + 'images/') 
