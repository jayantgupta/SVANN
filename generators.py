import keras
import numpy as np

from skimage.io import imread
from skimage.transform import resize

class Batch_Generator(keras.utils.Sequence):
  def __init__(self, image_filenames, mask_filenames, batch_size, image_height, image_width, image_channels) :
    self.image_filenames = image_filenames
    self.mask_filenames = mask_filenames
    self.batch_size = batch_size
    self.image_height = image_height
    self.image_width = image_width
    self.image_channels = image_channels
    
  def __len__(self) :
    return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int)
  
  
  def __getitem__(self, idx):
    image_batch_fns = self.image_filenames[idx * self.batch_size : (idx+1) * self.batch_size]
    mask_batch_fns = self.mask_filenames[idx * self.batch_size : (idx+1) * self.batch_size]
    
    image_batch_arrays = np.array([resize(imread(file_name), (self.image_height, self.image_width, self.image_channels)) for file_name in image_batch_fns])/255.
    mask_batch_arrays = np.array([resize(imread(file_name), (self.image_height, self.image_width, 1)) for file_name in mask_batch_fns])

    return image_batch_arrays, mask_batch_arrays

class Data_Generator(keras.utils.Sequence):
  def __init__(self, image_filenames, batch_size, image_height, image_width, image_channels, normalize) :
    self.image_filenames = image_filenames
    self.batch_size = batch_size
    self.image_height = image_height
    self.image_width = image_width
    self.image_channels = image_channels
    self.normalize = normalize
    
  def __len__(self) :
    return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int)
  
  
  def __getitem__(self, idx):
    batch_fns = self.image_filenames[idx * self.batch_size : (idx+1) * self.batch_size]
    
    batch_arrays = np.array([resize(imread(file_name), (self.image_height, self.image_width, self.image_channels)) for file_name in batch_fns])
    if self.normalize:
      batch_arrays = batch_arrays/255.

    return batch_arrays