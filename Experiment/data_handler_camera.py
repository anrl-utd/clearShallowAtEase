import cv2
import os
import glob
from sklearn.utils import shuffle
import numpy as np



def load_dataset(train_path, image_size, labels):
    images = [[] for x in range(243)]
    img_labels = [[] for x in range(243)]
    img_names = [[] for x in range(243)]
    cls = [[] for x in range(243)]
    person = 0
    car = 0
    print('Going to read training images')
    for label in labels:   
        index = labels.index(label)
        print(index)
        print('Now going to read {} files (Index: {})'.format(label, index))
        path = os.path.join(train_path, label, '*g')
        files = glob.glob(path)
        for fl in files:
            image = cv2.imread(fl)
            cam = int(fl[-16:-15])
            frame = int(fl[-7:-4])
            # already processed images, don't need to cheeck for 'None'
            #if image is None:
            #    continue
            # we assume the image size is 32x32
            #image = cv2.resize(image, (image_size, image_size),0,0, cv2.INTER_LINEAR)

            image = image.astype(np.float32)

            # normalize image for smaller range of pixel values [0,1]
            image = np.multiply(image, 1.0 / 255.0)
        
            # figure out a better way to deal with cls, img_names, img_labels without changing structure
            images[frame].append(image)
            label = np.zeros(len(labels))
            if index == 0:
                person += 1
            if index == 1:
                car += 1
            label[index] = 1.0
            # label is one-hot list
            img_labels[frame].append(label)
            flbase = os.path.basename(fl)
            img_names[frame].append(flbase)
            cls[frame].append(label)
    true_labels = img_labels
    print('Person files read in: ', person)
    print('Car files read in: ', car)
    # make every index of images have 6 images only (unroll)
    for i in range(len(images)):
        length = len(images[i])
        while length > 6 and length % 6 == 0:
            split_images = images[i][-6:]
            images[i] = images[i][:-6]
            images.append(split_images)
            
            split_labels = img_labels[i][-6:]
            #print(split_labels)
            img_labels.append(split_labels)
            img_labels[i] = img_labels[i][:-6]
            
            split_names = img_names[i][-6:]
            img_names[i] = img_names[i][:-6]
            img_names.append(split_names)

            cls.append(cls[i][-6])
            del cls[i][-6:]
            length = len(images[i])
            #print(len(images[i]), ' :', i)
    # now, combine images at each index of 'images' into a single image. we will split it in tf
    i = 0
    while i < len(images):
        len_imgs = len(images[i])
        # prune empty values of frames that may not be in the dataset (ie, 240 is in train, not in val)
        while len_imgs is not 6:
            del images[i]
            del img_names[i]
            del img_labels[i]
            del cls[i]
            len_imgs = len(images[i])
        #print(len(images[i]), ' :', i)
        #images[i] = np.concatenate((images[i][0],images[i][1],images[i][2],images[i][3],images[i][4],images[i][5]), axis = 1)
        i += 1
    non_one_hot = img_labels
    # make every element of img_labels a vec
    for l in range(len(img_labels)):
        if isinstance(img_labels[l], list):
            #print(img_labels[l])
            img_labels[l] = img_labels[l][0]
    car = 0
    person = 0
    #print(img_labels)
    
    for l in img_labels:
        if l[0] == 1:
            person += 1
        if l[1] == 1:
            car += 1
    bus = len(img_labels) - car - person
    print('Car: ' , car)
    print ('Person: ', person)
    print('Bus: ', bus)
    #print(img_labels)
    
    images = np.array(images)
    img_labels = np.array(img_labels)
    img_names = np.array(img_names)
    cls = np.array(cls)

    return images, img_labels, img_names, cls

def read_dataset(dataset_path, image_size, labels = ['person_images', 'car_images', 'bus_images']):

    images, img_labels, img_names, classes = load_dataset(dataset_path, image_size, labels)
    images, img_labels, img_names, classes = shuffle(images, img_labels, img_names, classes)

    return images, img_labels, img_names, classes

# used for just testing
if __name__ == "__main__":
    images, img_labels, img_names, classes = read_dataset("/Users/ashkany/Documents/GitHub/ResiliNet/multiview-dataset/test_dir", 32)
    print(images.shape)
