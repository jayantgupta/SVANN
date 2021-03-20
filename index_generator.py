# First, read the .tif files
# Then, generate x samples
# Use the samples to generate the Variogram

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageMath
from skgstat import Variogram, OrdinaryKriging
import os
import json

from tqdm import tqdm

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
    coords = generate_points_with_min_distance(n=30000, shape=sample_shape, min_dist=20)
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
    
    _, ax = plt.subplots(1, 1, figsize=(9, 9))
    art = ax.scatter(coords[:,0], coords[:,1], s=10, c=GRVI_array, cmap='plasma')
    plt.colorbar(art)
    plt.title("GRVI") 

    _, ax = plt.subplots(1, 1, figsize=(9, 9))
    art = ax.scatter(coords[:,0], coords[:,1], s=10, c=VARI_array, cmap='plasma')
    plt.colorbar(art)
    plt.title("VARI")
    
    _, ax = plt.subplots(1, 1, figsize=(9, 9))
    art = ax.scatter(coords[:,0], coords[:,1], s=10, c=ExG_array, cmap='plasma')
    plt.colorbar(art)
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
    plt.imshow(im)
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

def generate_index_channel(index_type,
                        in_dir_root=json.load(open('config.json'))['filepaths']['input_imagery_dir'], 
                        in_folder='',
                        out_dir = json.load(open('config.json'))['filepaths']['default_indices_dir'],
                        image_height=256,
                        image_width=256):
    current_dir = os.path.join(in_dir_root, in_folder) # in_dir_root/in_folder
    for filename in tqdm(os.listdir(current_dir)):
        in_file = os.path.join(current_dir, filename) # in_dir_root/in_folder/filename
        if os.path.isdir(in_file):
            print('\nProcessing /%s...'%(in_file))
            generate_index_channel(index_type, in_folder=os.path.join(in_folder, filename)) # Recursively call generate_index_channel for traversing folders
        if in_file.endswith(".tif") is not True:
            continue
        out_path = os.path.join(out_dir, os.path.join(in_folder, filename.split('.')[0])) # out_dir/in_folder/filename
        if not os.path.exists(os.path.join(out_dir, in_folder)): # Create out_dir/in_folder filepath
            os.makedirs(os.path.join(out_dir, in_folder))
        elif os.path.exists(out_path + "_"+index_type+".tif"): # Skipping files which have already been generated
            continue
        img = Image.open(in_file)
        img = img.resize((image_height, image_width)) # Preemptively resize channel to save time and storage space
        Band_B, Band_G, Band_R = img.split()

        # ImageMath expressions to calculate the RGB Index channels
        if index_type == "ExG":
            # Normalize RGB bands
            Band_B_n = ImageMath.eval("float(b)/float(r+g+b)", r=Band_R, g=Band_G, b=Band_B)
            Band_G_n = ImageMath.eval("float(g)/float(r+g+b)", r=Band_R, g=Band_G, b=Band_B)
            Band_R_n = ImageMath.eval("float(r)/float(r+g+b)", r=Band_R, g=Band_G, b=Band_B)
            index_array = ImageMath.eval("(2*g)-r-b", r=Band_R_n, g=Band_G_n, b=Band_B_n)
        elif index_type == "GRVI":
            index_array = ImageMath.eval("float(g-r)/float(g+r)", r=Band_R, g=Band_G, b=Band_B)
        elif index_type == "VARI":
            index_array = ImageMath.eval("float(g-r)/float(g+r-b)", r=Band_R, g=Band_G, b=Band_B)
            index_array = Image.fromarray(np.clip(np.asarray(index_array), -1, 1)) # Clip extremities
        elif index_type == "CIVE":
            index_array = ImageMath.eval("(0.441*float(r))-(0.811*float(g))+(0.385*float(b))+18.787", r=Band_R, g=Band_G, b=Band_B)

        index_array.save(out_path + "_"+index_type+".tif")

        #plt.subplots(1, 1, figsize=(9, 9))
        #art = plt.imshow(img)
        #plt.title("RGB")
        #
        #plt.subplots(1, 1, figsize=(9, 9))
        #art = plt.imshow(np.asarray(index_array), cmap ='plasma')
        #plt.colorbar(art)
        #plt.title(index_type)
        #plt.show()

if __name__ == '__main__':
    # read_file('datasets/MN_raster_Hennepin_North/120_23_13_01.tif')
    for index in tqdm(["VARI", "GRVI", "ExG", "CIVE"]):
        print('\nGenerating %s index channel...'%(index))
        generate_index_channel(index)
    
            
    