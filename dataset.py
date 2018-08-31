import cv2
import os
import glob
from sklearn.utils import shuffle
import numpy as np


def load_train(train_path, image_size, classes):
    images = [[] for x in range(243)]
    labels = [[] for x in range(243)]
    img_names = [[] for x in range(243)]
    cls = [[] for x in range(243)]
    person = 0
    car = 0
    print('Going to read training images')
    for fields in classes:   
        index = classes.index(fields)
        print(index)
        print('Now going to read {} files (Index: {})'.format(fields, index))
        path = os.path.join(train_path, fields, '*g')
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
        
            # figure out a better way to deal with cls, img_names, labels without changing structure
            images[frame].append(image)
            label = np.zeros(len(classes))
            if index == 0:
                person += 1
            if index == 1:
                car += 1
            label[index] = 1.0
            # label is one-hot list
            labels[frame].append(label)
            flbase = os.path.basename(fl)
            img_names[frame].append(flbase)
            cls[frame].append(fields)
    true_labels = labels
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

    return shuffle(self._images[start:end], self._labels[start:end], self._img_names[start:end], self._cls[start:end])


def read_train_sets(train_path, val_path, image_size, classes):
  class DataSets(object):
    pass
  data_sets = DataSets()

  train_images, train_labels, train_img_names, train_cls = load_train(train_path, image_size, classes)
  train_images, train_labels, train_img_names, train_cls = shuffle(train_images, train_labels, train_img_names, train_cls)
  
  validation_images, validation_labels, validation_img_names, validation_cls = load_train(val_path, image_size, classes) 
  validation_images, validation_labels, validation_img_names, validation_cls = shuffle(validation_images, validation_labels, validation_img_names, validation_cls) 

 # we modify the way the validation set is handled, wasn't actually coming from "/testing_data/"
  ''' 
 if isinstance(validation_size, float):
    validation_size = int(validation_size * images.shape[0])

  validation_images = images[:validation_size]
  validation_labels = labels[:validation_size]
  validation_img_names = img_names[:validation_size]
  validation_cls = cls[:validation_size]

  train_images = images[validation_size:]
  train_labels = labels[validation_size:]
  train_img_names = img_names[validation_size:]
  train_cls = cls[validation_size:]
  '''
  data_sets.train = DataSet(train_images, train_labels, train_img_names, train_cls)
  data_sets.valid = DataSet(validation_images, validation_labels, validation_img_names, validation_cls)

  return data_sets
