import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from matplotlib.widgets import Cursor
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter
from scipy.ndimage import zoom
from skimage import io
from skimage.transform import resize
import cv2
import mrcfile
import open3d as o3d
from mayavi import mlab
from mayavi.core.ui.api import MlabSceneModel, SceneEditor, MayaviScene
from traits.api import HasTraits, Instance, Button, on_trait_change
from traitsui.api import View, Item, HSplit, VSplit, Group
import os
import shutil
import glob

from .public_imagesManipulation import downscale_img
from .public_filesManipulation import *

def plots(data, angle, fif_metric, curv_metric, path, colormap='rainbow'):
    '''
    Plots the data in 9 subplots.
    Parameters: - data: array containig the data (images) for each subplot.
                        It is assumed to be of length 7 or 9. First image
                        is raw data, 2nd is empty, the fitness of fit (NELD) 3rd, the
                        global fitness of fit 4, the reconstruction 5 and
                        the reconstruction with interpolation of non-processed pixels 6,
                        and 7, 8, 9 and 10correspond to the curvature.
                        If curv_metric is 'gaussian' or 'mean, then image 7 is the
                        curvature of the reconstruction and 8 the curvature of the
                        interpolated reconstruction. If curv_metric is 'principal',
                        7 and 8 correspond to the recontruction curvature and
                        9 and 10 to the curvature of the interpolation.
                - angle: first angle in degrees for which the data was taken.
                - fif_metric: string,
                - curv_metric: metric used to calculate curvature;
                               options: 'gaussian', 'mean', 'principal';
                               principal is a tuple (principal_max, principal_min).
                - path: location to save plots.
                - colormap: LUT for images 1 to 5 in data.
    Returns: None
    '''
    print(len(data))
    font = 'Arial'
    font_props = font_manager.FontProperties(family='Arial', \
                                                          style='normal', \
                                                          size=15)

    # plot raw image at first angle
    fig, axes = plt.subplots(5, 2, figsize=(20,20))
    axes = axes.flatten()
    title_size = 20
    # plot raw image at first angle
    sub_fig = 0
    img = data[sub_fig]
    bar = axes[sub_fig].imshow(img, cmap=colormap)
    _ = axes[sub_fig].set_title(f"Raw itensity image at angle {round(angle, 2)}°", size=title_size, fontname=font)
    _ = axes[sub_fig].axis("off")
    _ = bar.set_clim(np.min(img[img > 0]), np.max(img))
    _ = fig.colorbar(bar, ax=axes[sub_fig])
    #bar.set_clim(minc, maxc)

    # delete subfigure area
    sub_fig = 1
    fig.delaxes(axes[sub_fig])

    # plot fitnes of fit image
    sub_fig = 2
    img = data[sub_fig - 1]
    barfif = axes[sub_fig].imshow(img, cmap=colormap)
    _ = axes[sub_fig].set_title("NELD", size=title_size, fontname=font)
    _ = axes[sub_fig].axis("off")
    #_ = barfif.set_clim(np.min(img[img != 0]), np.max(img))
    _ = fig.colorbar(barfif, ax=axes[sub_fig])

    # plot global fitnes of fit image
    sub_fig = 3
    img = data[sub_fig - 1]
    barfif = axes[sub_fig].imshow(img, cmap=colormap)
    _ = axes[sub_fig].set_title("Global NELD", size=title_size, fontname=font)
    _ = axes[sub_fig].axis("off")
    #_ = barfif.set_clim(np.min(img[img != 0]), np.max(img))
    _ = fig.colorbar(barfif, ax=axes[sub_fig])

    # plot reconstruction image
    sub_fig = 4
    img = np.around(data[sub_fig - 1] / 1000, 3)
    bar = axes[sub_fig].imshow(img, cmap=colormap)
    _ = axes[sub_fig].set_title("Height (μm)", size=title_size, fontname=font)
    _ = axes[sub_fig].axis("off")
    #_ = bar.set_clim(np.min(img[img != 0]), np.max(img))
    _ = fig.colorbar(bar, ax=axes[sub_fig])

    # plot interpolated reconstruction image
    sub_fig = 5
    img = np.around(data[sub_fig - 1] / 1000, 3)
    bar = axes[sub_fig].imshow(img, cmap=colormap)
    _ = axes[sub_fig].set_title("Height (μm) (interp)", size=title_size, fontname=font)
    _ = axes[sub_fig].axis("off")
    #_ = bar.set_clim(np.min(img[img != 0]), np.max(img))
    _ = fig.colorbar(bar, ax=axes[sub_fig])

    if (curv_metric != 'principal'):
        # plot curvature and save image
        sub_fig = 6
        img = data[sub_fig - 1]
        barcurv = axes[sub_fig].imshow(img, cmap=colormap)
        _ = axes[sub_fig].set_title("Curvature: " + curv_metric, size=title_size, fontname=font)
        _ = axes[sub_fig].axis("off")
        #_ = barcurv.set_clim(np.min(img[img != 0]), np.max(img))
        _ = fig.colorbar(barcurv, ax=axes[sub_fig])

        sub_fig = 7
        img = data[sub_fig - 1]
        barcurv = axes[sub_fig].imshow(img, cmap=colormap)
        _ = axes[sub_fig].set_title("Curvature (interp): " + curv_metric, size=title_size, fontname=font)
        _ = axes[sub_fig].axis("off")
        #_ = barcurv.set_clim(np.min(img[img != 0]), np.max(img))
        _ = fig.colorbar(barcurv, ax=axes[sub_fig])

        # delete subfigure area
        sub_fig = 8
        fig.delaxes(axes[sub_fig])
        sub_fig = 9
        fig.delaxes(axes[sub_fig])


    else:
        # plot principle_max reconstruction
        sub_fig = 6
        img = data[sub_fig - 1]
        barcurv = axes[sub_fig].imshow(img, cmap=colormap)
        _ = axes[sub_fig].set_title("Curvature: principal maximum", size=title_size, fontname=font)
        _ = axes[sub_fig].axis("off")
        #_ = barcurv.set_clim(np.min(img[img != 0]), np.max(img))
        _ = fig.colorbar(barcurv, ax=axes[sub_fig])

        # plot principle_min reconstruction
        sub_fig = 7
        img = data[sub_fig - 1]
        barcurv = axes[sub_fig].imshow(img, cmap=colormap)
        _ = axes[sub_fig].set_title("Curvature: principal minimum", size=title_size, fontname=font)
        _ = axes[sub_fig].axis("off")
        #_ = barcurv.set_clim(np.min(img[img != 0]), np.max(img))
        _ = fig.colorbar(barcurv, ax=axes[sub_fig])

        # plot principle_max reconstruction
        sub_fig = 8
        img = data[sub_fig - 1]
        barcurv = axes[sub_fig].imshow(img, cmap=colormap)
        _ = axes[sub_fig].set_title("Curvature (interp): principal maximum", size=title_size, fontname=font)
        _ = axes[sub_fig].axis("off")
        #_ = barcurv.set_clim(np.min(img[img != 0]), np.max(img))
        _ = fig.colorbar(barcurv, ax=axes[sub_fig])

        # plot principle_min reconstruction
        sub_fig = 9
        img = data[sub_fig - 1]
        barcurv = axes[sub_fig].imshow(img, cmap=colormap)
        _ = axes[sub_fig].set_title("Curvature (interp): principal minimum", size=title_size, fontname=font)
        _ = axes[sub_fig].axis("off")
        #_ = barcurv.set_clim(np.min(img[img != 0]), np.max(img))
        _ = fig.colorbar(barcurv, ax=axes[sub_fig])

    # Save plots as an image
    plt.savefig(path)
    plt.savefig(path[:-3] + 'svg')
    #plt.show()
