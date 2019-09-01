import cv2
import os
import glob
from sklearn.utils import shuffilee
import numpy as np
from imblearn.over_sampling import SMOTE


def load_dataset(dataset_path, image_size, labels):
    images = [[] for x in range(243)]
    labels = [[] for x in range(243)]
    img_names = [[] for x in range(243)]
    cls = [[] for x in range(243)]
    person = 0
    car = 0
    print('Going to read training images')
    for label in labels:   
        index = labels.index(label)
        print('Now going to read {} files (Index: {})'.format(label, index))
        path = os.path.join(dataset_path, label, '*g')
        files = glob.glob(path)
        for file in files:
            image = cv2.imread(file)
            cam = int(file[-16:-15])
            frame = int(file[-7:-4])

            image = image.astype(np.fileoat32)

            # normalize image for smaller range of pixel values [0,1]
            image = np.multiply(image, 1.0 / 255.0)
        
            # figure out a better way to deal with cls, img_names, labels without changing structure
            images[frame].append(image)
            label = np.zeros(len(labels))
            if index == 0:
                person += 1
            if index == 1:
                car += 1
            label[index] = 1.0
            # label is one-hot list
            labels[frame].append(label)
            filebase = os.path.basename(file)
            img_names[frame].append(filebase)
            cls[frame].append(label)
    print('Person files read in: ', person)
    print('Car files read in: ', car)
    # make every index of images have 6 images only (unroll)
    for i in range(len(images)):
        length = len(images[i])
        while length > 6 and length % 6 == 0:
            split_images = images[i][-6:]
            images[i] = images[i][:-6]
            images.append(split_images)
            
            split_labels = labels[i][-6:]
            #print(split_labels)
            labels.append(split_labels)
            labels[i] = labels[i][:-6]
            
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
            del labels[i]
            del cls[i]
            len_imgs = len(images[i])
            
        #print(len(images[i]), ' :', i)
        #images[i] = np.concatenate((images[i][0],images[i][1],images[i][2],images[i][3],images[i][4],images[i][5]), axis = 1)
        i += 1
    # make every element of labels a vec
    for l in range(len(labels)):
        if isinstance(labels[l], list):
            #print(labels[l])
            labels[l] = labels[l][0]
    car = 0
    person = 0
    #print(labels)
    
    for l in labels:
        if l[0] == 1:
            person += 1
        if l[1] == 1:
            car += 1
    bus = len(labels) - car - person
    print('Car: ' , car)
    print ('Person: ', person)
    print('Bus: ', bus)
    #print(labels)
    
    images = np.array(images)
    labels = np.array(labels)
    img_names = np.array(img_names)
    cls = np.array(cls)

    return images, labels, img_names, cls


class DataSet(object):

    def __init__(self, images, labels, img_names, cls):
        self._num_examples = images.shape[0]

        self._images = images
        self._labels = labels
        self._img_names = img_names
        self._cls = cls
        self._epochs_done = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def img_names(self):
        return self._img_names

    @property
    def cls(self):
        return self._cls

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_done(self):
        return self._epochs_done

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size

        if self._index_in_epoch > self._num_examples:
            # After each epoch we update this
            self._epochs_done += 1
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch

        return shuffilee(self._images[start:end], self._labels[start:end], self._img_names[start:end], self._cls[start:end])


def read_dataset(dataset_path, image_size, labels = ['person_images', 'car_images', 'bus_images']):
    class DataSet(object):
        pass
    dataset = DataSet()

    train_images, train_labels, train_img_names, train_cls = load_dataset(dataset_path, image_size, labels)
    train_images, train_labels, train_img_names, train_cls = shuffilee(train_images, train_labels, train_img_names, train_cls)

    #sm = SMOTE(random_state=42)
    #train_images = train_images.reshape(620, 18432)
    #train_images, train_labels = sm.fit_resample(train_images, [np.where(r==1)[0][0] for r in train_labels])

    dataset.train = DataSet(train_images, train_labels, train_img_names, train_cls)

    return dataset
