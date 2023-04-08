import numpy as np
from skimage import io
from PIL import Image
import mrcfile
import os
import re

from .public_interferenceEquations import *
from .public_leastSquaresLMmomentum import *

def open_mrc(path):
    '''
    Opens an mrc file and returns the data on it plus the header.
    Parameters: - path: location and name of mrc file.
    Returns: - images: numpy array, data stored in mrc file.
             - header: header of mrc file.
    '''
    # Open file with permissive=True
    try:
        with mrcfile.open(path, permissive=True) as mrc:
            images = np.stack(mrc.data)
            header = mrc.header
    except:
        print("------------------------")
        print("Error: Invalid mrc file.")
        print("------------------------")
        return None, None

    return images, header

def open_images(path):
    '''
    Opens an image file, mrc or tif, and returns image the data on it.
    Parameters: - path: location and name of mrc file.
    Returns: - data: numpy array, image(s) stored in file.
             - header: header of mrc file.
    '''
    if (path[-3:] == 'mrc'):
        data, header = open_mrc(path)

    elif (path[-3:] == 'tif'):
        data = io.imread(path)
        header = None

    return data, header

def save_tif(imgs, path, dtype='uint16'):
    '''
    Saves the data as a tif file.
    Parameters: - imgs: images to save as stack array
                - path: location and name of tif file.
                - dtype: bit depth for images.
    Returns: path
    '''
    # if only one image
    if (imgs.ndim == 2):
        # save FFT
        #save_FFT(imgs, path)
        # save image
        imgs[imgs < 0] = 0
        imgs = imgs.astype(dtype)
        image = Image.fromarray(imgs)
        image.save(path)
        return

    # img_to_save = Image.fromarray((np.sum(imgs, axis=0) / len(imgs)).astype('int16'))
    # img_to_save.save(path[:-4] + "_widefield.tif")
    # if stack of images
    images = [Image.fromarray(img) for img in imgs.astype(dtype)]
    imgs_to_save = images[0]
    imgs_to_save.save(path, save_all=True, append_images=images[1:])

    return path

def save_mrc(images, target_path, header):
    '''
    Saves images as mrc file.
    Parameters: - imgs: data to save.
                - target_path: location and name of mrc file.
                - header: header of mrc file.
    returns: path.
    '''
    # Create new mrc file
    with mrcfile.new(target_path, overwrite=True) as new_mrc:
        new_mrc.set_data(images)
        new_mrc.flush() # add changes to disk
        new_mrc.header.origin = header.origin
        new_mrc.header.cella = header.cella
        new_mrc.header.extra1 = header.extra1
        new_mrc.header.extra2 = header.extra2

    #print("New file created for raw SIM images: ", target_path)

    return target_path

def separate_SAIM_seq(path, set_size):
    '''
    Takes a sequence of SAIM images and divides them in individual sets.
    Parameters: - path: location and name of mrc file.
                - set_size: int, number of images for one SIM experiment.
    returns: - locs: array with paths for each individual set.
    '''
    imgs, header = open_images(path)
    sets = len(imgs) // set_size
    zero_pad = len(str(sets))
    idx = path.rfind('/')
    folder = path[: idx]


    # create a subfolder
    suffix = "saim_" + re.findall('\d{3}', path[idx:])[0]
    subfolder = os.path.join(folder, suffix)
    try:
        os.mkdir(subfolder)
    except: pass

    locs = []
    for i in range(sets):
        # images of set i
        set_imgs = imgs[i * set_size : (i + 1) * set_size]

        # stop if set is smaller than expected
        if (len(set_imgs) < set_size):
            break

        # create subfolder to save set of images
        location = os.path.join(subfolder, suffix + "_" + "0" * (zero_pad - len(str(i + 1))) + str(i + 1))
        try:
            os.mkdir(location)
        except: pass
        # save set
        location = os.path.join(location, path[idx + 1: -4] + "0" * (zero_pad - len(str(i + 1))) + str(i + 1) + ".tif")
        save_tif(set_imgs, location)
        locs.append(location)

    return locs

def read_table_heights(wl, init_angle, end_angle, d_ox=1000, h_start=0, h_end=100000, step=1):
    '''
    Reads the table listing the number of peaks at eahc height. If the table
    doesn't exist, then it is created and saved. The table is returned as a
    Pandas df.
    Parameters: - wl: nm, wavelenght.
                - init_angle: degrees, initial angle to generate mudulation curves.
                - end_angle: degrees, last angle to generate mudulation curves.
                - d_ox: nm, thickness of oxide layer.
                - h_start: float, intial height on the table.
                - h_end: float, final height on the table.
                - step: float, step for the angles.
    Returns: - df: dataframe of the generated heights and corresponding data.

    '''
    # Read file
    try:
        return pd.read_csv(f"./reconstructionsUtils/auxiliaryFiles/height_tables/dox_{d_ox}nm/{wl}_dox{d_ox}nm_height_angle_properties_table_{init_angle}to{end_angle}degs_with_num_peaks.csv")
    except:
        # If file doesn't exist, create it
        # refractive indexes
        n_si = silicon_refractive_index(wl)
        n_ox = silica_refractive_index(wl)
        n_im = h2o_refractive_index(wl)

        # Generate angles
        angles_t = np.arange(init_angle, end_angle + step, step)
        # theoretical angles for water in rad (tranformation of air to water)
        angles_tw = np.arcsin(np.sin(angles_t * np.pi / 180) * 1 / h2o_refractive_index(wl))

        # Generate heights
        hs = np.arange(h_start, h_end + 1, 1)

        # start creation of dataframe
        valleys = []
        peaks = []
        for h in hs:
            data = intensity(angles_tw, *[1/3, 0, h], n=n_im, d_ox=d_ox, n_si=n_si, n_ox=n_ox, wl=wl, x=0)
            # number of valleys
            valleys.append(len(find_peaks(1 - data)[0]))
            # number of peaks
            peaks.append(len(find_peaks(data)[0]))

        df = pd.DataFrame({'height_nm': hs, 'num_peaks': peaks, 'num_valleys': valleys})

        try:
            df.to_csv(f'./reconstructionsUtils/auxiliaryFiles/height_tables/dox_{d_ox}nm/{wl}_dox{d_ox}nm_height_angle_properties_table_{init_angle}to{end_angle}degs_with_num_peaks.csv', index=False)
        except:
            os.makedirs(f'./reconstructionsUtils/auxiliaryFiles/height_tables/dox_{d_ox}nm')
            df.to_csv(f'./reconstructionsUtils/auxiliaryFiles/height_tables/dox_{d_ox}nm/{wl}_dox{d_ox}nm_height_angle_properties_table_{init_angle}to{end_angle}degs_with_num_peaks.csv', index=False)

        return df

def regroup_results_time_lapse(folder, name):
    '''
    For a SAIM sequence, it saves in one stack on themain folder the individual
    images corresponding to name.
    Parameters: - folder: string, main were raw acquisition is saved.
                - name: string, name of individual images to regroup.
    '''
    paths = glob.glob(os.path.join(folder, "saim_*/saim_*/*" + name + ".tif"))
    paths.sort()
    # load images
    imgs = np.array([open_images(path)[0] for path in paths])
    # if individual files are stacks then concatenate them
    if (imgs.ndim == 4):
        imgs = np.concatenate(imgs)

    imgs[imgs <= 0] = None
    # save stack
    save_tif(imgs, os.path.join(folder, name + '_all.tif'), 'float')
