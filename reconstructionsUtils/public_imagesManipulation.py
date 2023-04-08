import numpy as np
from skimage.transform import resize


def downscale_img(img, factor=2):
    '''
    Downscales image by the given factor with anti-aliasing.
    Parameters: - img: image to downscale.
                - factor: downscaling factor, default reduces image by 2.
    Returns: - new_img: downscaled image.
    '''
    new_dims = np.array(img.shape) // 2
    new_img = resize(img, new_dims, anti_aliasing=True)

    return new_img

def correct_intensity_images(images, cor_images):
    '''
    Correct intensity of images using data in cor_images.
    Parameters: - images: numpy array of images to correct.
                - cor_images: numpy array of images used fo correction.
                              It is assumed that the images in cor_images
                              where taken from a uniform fluorescent slide
                              or a dye solution.
    Returns: numpy array with corrected images.
    '''
    if (len(images) != len(cor_images)):
        print("--------------------------------------------------------------------")
        print("Error: number of correction images don't match number of SAIM images")
        print("--------------------------------------------------------------------")
        return None

    cor_images = np.divide(cor_images[0], cor_images)

    return np.multiply(images, cor_images)

##################################
## Power compensation on images ##
##################################

def power_compensation(data):
    '''
    Applies weights to the data to compensate for differences in laser power
    between acquisitions at different anfgles. The weights were determined
    exprimentally.
    Parameters:
        data: 1D or 2D np array, detected emission.
    Returns: corected data.
    '''
    weights = [1, 1, 1, 1, 1, 1, 1, 1, 1, 0.64/0.63, 0.64/0.63, 0.64/0.63, 0.64/0.63, \
               0.64/0.62, 0.64/0.61, 0.64/0.61, 0.64/0.6, 0.64/0.6, 0.64/0.6, 0.64/0.6, \
               0.64/0.59, 0.64/0.59, 0.64/0.58, 0.64/0.58, 0.64/0.57, 0.64/0.57, \
               0.64/0.56, 0.64/0.56, 0.64/0.55, 0.64/0.54]

    weights = np.array(weights)

    # When data is a 1D array
    if (np.ndim(data) == 1):
        return np.multiply(data, weights)

    # When data is a stack of images
    elif (np.ndim(data) == 3):
        return np.array([img * weight for img, weight in zip(data, weights)])

def interpolate_pixel(image, upper_limit=np.float('inf'), lower_limit=0):
    '''
    Modifies image by changing the value of a negative pixel to the average of
    its non-negative neighbors.
    Parameters: - image: numpy 2D array
    Returns: image with non-negative pixels.
    '''
    image[np.isnan(image)] = -1
    # Change each negative pixel
    for y, x in zip(*((image < lower_limit) | (image >= upper_limit)).nonzero()):
        # limits for neighbors locations
        ymin = (y - 1) * ((y - 1) > 0) # ReLu implementation
        ymax = y + 2
        xmin = (x - 1) * ((x - 1) > 0) # ReLu implementation
        xmax = x + 2
        # central pixel + neighbors
        neighbors = image[ymin:ymax, xmin:xmax]
        # assign mean value to central pixel
        image[y, x] = neighbors[(neighbors >= lower_limit) & (neighbors < upper_limit)].mean()

    #return image

def sort_angles_match_images(angles, data):
    '''
    Sorts angles list in increasing order and matches each element in data to each element in angles.
    If angles is sorted, the function returns the parameters unchanged.
    Parameters: - angles: array of angles.
                - data: array of data where each element's index position corresponds to the angle
                        at the same index in angles.
    Returns: - sorted_angles: sorted list.
             - sorted_data: images matched to respective angle.
    '''
    # if angles is sorted, no need to modify
    if (np.sum(angles == np.sort(angles)) == len(angles)):
        return angles, data

    angle_dict = dict(zip(angles, data))
    sorted_angles = np.sort(list(angle_dict.keys()))
    sorted_data = [angle_dict[ang] for ang in angle_dict.keys()]
    sorted_data = np.array(sorted_data)

    return sorted_angles, sorted_data

def apply_fif_threshold(image, fif_img, thresh):
    '''
    This function creates a new image of the height reconstruction
    by applying the threshold in the fitness of fit image.
    Parameters: - image: np.array, reconstruction image.
                - fif_img: np.array, fitness of fit image.
                - thresh: float, threshold to apply.
    Returns: modifed image.
    '''
    image[fif_img > thresh] = None
    # change value of negative pixels to average of positive neighbors
    _ = interpolate_pixel(image)

    return image
