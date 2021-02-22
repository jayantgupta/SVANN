# First, read the .tif files
# Then, generate x samples
# Use the samples to generate the Variogram

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skgstat import Variogram, OrdinaryKriging
import os

os.environ["SKG_SUPPRESS"] = "true"

Image.MAX_IMAGE_PIXELS = 219494175
#Image.MAX_IMAGE_PIXELS = 114779595

def GRVI(Band_R, Band_G):
    return float((Band_G - Band_R)*100) / (Band_G + Band_R)
#     return float(Band_R/256)

def VARI(Band_R, Band_G, Band_B):
    return float(((Band_G - Band_R)*100) / (Band_G + Band_R - Band_B + 0.01))

def ExG(Band_R, Band_G, Band_B):
    return float((2*Band_G - (Band_R + Band_B)) / (Band_G + Band_R + Band_B))*100
    
def read_file(path):
    im  = Image.open(path)
#    im_crop = im.crop((4096, 4096, 4096 + 2048, 4096 + 2048))
#    im = im_crop
    print(im.size, im.mode, im.format)
    sample_shape = (im.size[0], im.size[1])

    # Sampling.
    coords = generate_points_with_min_distance(n=3000, shape=sample_shape, min_dist=20)
    GRVI_array = np.zeros(shape = (len(coords), 1))
    VARI_array = np.zeros(shape = (len(coords), 1))
    ExG_array = np.zeros(shape = (len(coords), 1))
    # Calculating GRVI values.
    for ptr in range(len(coords)):
	# converting numpy.int64 to int.
        Band_R, Band_G, Band_B = im.getpixel((int(coords[ptr][0]), int(coords[ptr][1]))) 
        GRVI_array[ptr] = GRVI(Band_R, Band_G)
        VARI_array[ptr] = VARI(Band_R, Band_G, Band_B)
        ExG_array[ptr] = ExG(Band_R, Band_G, Band_B)
    
    print(coords.shape)
    
    ''' 
    fig, ax = plt.subplots(1, 1, figsize=(9, 9))
    art = ax.scatter(coords[:,0], coords[:,1], s=10, c=GRVI_array, cmap='plasma')
    plt.colorbar(art);
    plt.title("GRVI") 

    plt.figure()
    fig, ax = plt.subplots(1, 1, figsize=(9, 9))
    art = ax.scatter(coords[:,0], coords[:,1], s=10, c=VARI_array, cmap='plasma')
    plt.colorbar(art);
    plt.title("VARI")
    '''
    
    fig, ax = plt.subplots(1, 1, figsize=(9, 9))
    art = ax.scatter(coords[:,0], coords[:,1], s=10, c=ExG_array, cmap='plasma')
    plt.colorbar(art);
    plt.title("ExG")

    # Computing the variogram here.
    V = Variogram(coords, ExG_array.flatten(), model='spherical', n_lags=15, use_nugget=True)

    # Performing ordinary kriging here.
#    ok = OrdinaryKriging(V, min_points = 5, max_points = 15, mode='exact')
#    plt.figure()
    V.plot()
    print(V)
    im = im.rotate(90)
        
    # plot
    plt.figure()	
    implot = plt.imshow(im)
    plt.scatter(coords[:,0], coords[:,1], s=3)
    plt.show()

# Code to generate point samples from the image dataset.    
def generate_points_with_min_distance(n, shape, min_dist):
    # compute grid shape based on number of points
    width_ratio = shape[1] / shape[0]
    num_y = np.int32(np.sqrt(n / width_ratio)) + 1
    num_x = np.int32(n / num_y) + 1

    # create regularly spaced neurons
    x = np.linspace(0., shape[1]-1, num_x, dtype=np.float32)
    y = np.linspace(0., shape[0]-1, num_y, dtype=np.float32)
    coords = np.stack(np.meshgrid(x, y), -1).reshape(-1,2)
#    print(coords)

    # compute spacing
    init_dist = np.min((x[1]-x[0], y[1]-y[0]))

    # perturb points
    max_movement = (init_dist - min_dist)/2
    noise = np.random.uniform(low=-max_movement,
                                high=max_movement,
                                size=(len(coords), 2))
    coords += noise
    mask = (coords[:,0] > 0) & (coords[:,0] < shape[0]) & (coords[:,1] > 0) & (coords[:,1] < shape[1])
#    print (mask)
#    print(coords[mask])
    return (coords[mask].astype(int))

if __name__ == '__main__':
#    read_file('/Users/jayantgupta/Desktop/SV/Tiles/027_23_06_01_tile_0-0.tif')
#    read_file('/Users/jayantgupta/Desktop/SV/Tiles/027_23_06_01_tile_0-4936.tif')
#    read_file('/Users/jayantgupta/Desktop/SV/Tiles/027_23_06_01_tile_5340-0.tif')
    read_file('/Users/jayantgupta/Desktop/SV/Tiles/027_23_06_01_tile_5340-4936.tif')
#    read_file('/Users/jayantgupta/Desktop/SV/Tiles/027_23_06_01_tile_5340-4936.tif')
#    read_file('/Users/jayantgupta/Desktop/SV/Tiles/027_23_06_02_tile_5340-0.tif')
