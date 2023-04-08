import matplotlib
matplotlib.use('TkAgg')
import numpy as np
# import pandas as pd
import math
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib
import matplotlib.pyplot as plt
import time
from matplotlib import ticker
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from tkinter import *
from skimage import io
import cv2
import glob

from .public_interferenceEquations import *
from .public_leastSquaresLMmomentum import *
from .public_evaluationFunctions import *
from .public_filesManipulation import *
from .public_visualizationSave import *
from .public_imagesManipulation import *
from .public_reconstructions import *

def fill_reading_space(identifier, value):
    '''
    Add a string value to the location given by identifier.
    value is a string
    '''
    # activate field
    identifier.configure(state='normal')
    identifier.delete(0, END)
    identifier.insert(END, value)
    identifier.configure(state='readonly')


def num_pixels_to_analize(pixels, pix_op):
    '''
    Add values to pix_op.
    '''
    values = []
    for i in range(len(pixels)):
        values.append(str(i + 1))

    pix_op.delete(0, END)
    pix_op['values'] = values
    if (len(values) > 0):
        pix_op.current(0) #set the selected item

def add_height(row):
    '''
    Adds height to tracking data frame.
    '''
    return reconstructions[(row.frame - 1) // 69, int(row.y), int(row.x)]


def clicked_reconstruction_util(wave, dox, min_theta, max_theta, step_theta, pix_s, num_ap, index_nim, index_nox, index_nsi, \
                                fold, prosaim_file, roi_t, critical_angle, \
                                sbg_select, bg_pixs, \
                                sq_num, rec_time, h_mean, \
                                mean_neld, inith_seq_select, optional_h_window):
    '''
    This the function calling the revonstruction functions when
    the Update button in the GUI is pressed or the options for
    Fresnel Coefficient, Phase and PI Shift are changed.
    '''

    wl = int(wave.get())
    d_ox = int(dox.get())
    min_deg = float(min_theta.get())
    max_deg = float(max_theta.get())
    step_deg = float(step_theta.get())
    pixel_size = float(pix_s.get())
    na = float(num_ap.get())
    n_im = float(index_nim.get())
    n_ox = float(index_nox.get())
    n_si = float(index_nsi.get())
    folder = fold.get()
    saim_file = prosaim_file.get()

    # perform SAIM
    if (saim_file == ""):
        return

    # check ROI type
    roi_type = roi_t.get()
    if (roi_type == 'rectangular'):
        roi_type = 'rect'
    else:
        roi_type = 'polys'

    # look for ROI definition if present
    to_hFile = os.path.join(folder, f"height_rec.tif")
    try:
        roi = os.path.join(folder, "height_rec_roi_definition.npy")
        roi = roi.replace("//", "/")
        roi = np.load(roi, allow_pickle=True)
        roi_type = 'rect'
        roi_t.current(0)
    except:
        try:
            roi = os.path.join(folder, "height_rec_polygon_roi_definition.npy")
            roi = roi.replace("//", "/")
            roi = np.load(roi, allow_pickle=True)
            roi_type = 'polys'
            roi_t.current(1)
        except:
            roi = []

    # Calculate critical angle with respect to sample refractive index
    try:
        critical_angle_rad = math.asin(n_cell / n)
    except:
        critical_angle_rad = np.nan
    fill_reading_space(critical_angle, '%.2f'%(critical_angle_rad * 180 / np.pi))

    # Angles in degrees
    angles_deg = np.arange(min_deg, max_deg + step_deg, step_deg)
    # Angles in water
    angles_deg_w = np.arcsin(np.sin(angles_deg * np.pi / 180) * 1 / n_im) * 180 / np.pi
    min_rad = min_deg * np.pi / 180
    max_rad = max_deg * np.pi / 180
    # First and last angle in water
    angle_range_deg_w = (angles_deg_w[0], angles_deg_w[-1])

    # option for 3D figure
    option = 1
    # option to minimize NELD per pixel
    minimize_neld = True
    # nm, step for NELD minimization
    h_step = 100
    # minimum number of modulations
    group_size = 2
    # minimal relative amplitude with respect to peak to peak mean
    # to consider peak
    modulation_thresh = 0.2
    space_modulation_tresh = 0.6
    # option to use Scipy's least squares is ignored if GPU in the system
    scipy_fit = True
    # option to subtract background
    subtract_bg = sbg_select.get()
    # average of the lowest specified intensity values in ROI
    try:
        num_bg_vals = int(bg_pixs.get())
    except:
        # if empty string in bg_pixs
        num_bg_vals = None

    # option to freeze amplitude and bias to fit only with respect to height
    freeze_params = False
    # option to perform a second pass
    second_pass = inith_seq_select.get()
    # option to display 3D reconstructions
    vis3D = False
    # option to normalize data for each pixel
    normalization = True
    norm_type = 'minmax' #'max'
    # fitness of fit metric to use
    fif_metric = peakloc_diff # peakloc_diff is NELD #  chi_peak_combined # rmsea #
    # criteria to initialize height
    proper = 'minmax_diff'#'max_angle'
    # option to smooth data per pixel
    apply_filt = False
    filt_type = 'median'
    filt_window = 3
    savgol_poly = 7
    # option to compensate for laser power differences
    power_correction = False
    # option to interpolate value of negative pixels using mean of positive neighbors
    interpolate_neg = False
    # Choose to correct images for intensity differences using reference images
    correction = False
    # Choose to use number of peaks in criterion for initialization
    use_peaks = True
    # option to avoid fit and return the initialization values
    keep_initial_values = False
    # intialization (not used when minimizing NELD)
    params = np.array([1/3, 0, 6900])

    # path containing images to correct for illumination unevennes or changes in
    # laser power as the angle changes
    path_correction = ""
    # path with data to reconstruct
    org_path = os.path.join(folder, saim_file)
    # location to save reconstruction and associated files
    hImage_path = os.path.join(folder, to_hFile)

    # check if SAIM file is a time lapse experiment or not
    experiment_imgs, _ = open_images(org_path)
    time_lapse = (len(experiment_imgs) > len(np.arange(min_deg, max_deg + step_deg, step_deg)))

    # launch reconstruction
    if (time_lapse == False):
        # indicate sequence experiment number, in this case 1 of 1
        fill_reading_space(sq_num, '1 / 1')

        # check if there is an existing reconstruction
        try:
            previous_rec, _ = open_images(os.path.join(folder, height_rec2.tif))
        except:
            try:
                previous_rec, _ = open_images(os.path.join(folder, height_rec.tif))
            except:
                previous_rec = None

        start = time()
        SAIMimg_path, org, org_minus_bg, rec, neld_img, global_nelds_img = SAIM_reconstruction(org_path, hImage_path, \
                                       angles=angles_deg_w, angle_range=angle_range_deg_w, params=params, heights_matrix= previous_rec, \
                                       path_correction= path_correction, scipy_fit=scipy_fit, \
                                       subtract_bg=subtract_bg, roi=roi, roi_type=roi_type, \
                                       num_bg_vals=num_bg_vals, proper=proper, \
                                       keep_initial_values=keep_initial_values, use_peaks = use_peaks, \
                                       freeze_params=freeze_params, second_pass=second_pass, \
                                       power_correction=power_correction, \
                                       normalization=normalization, norm_type=norm_type, \
                                       correction=correction, interpolate_neg=interpolate_neg, \
                                       fif_metric=fif_metric, \
                                       minimize_rmse=minimize_neld, h_step=h_step, \
                                       group_size=group_size, modulation_thresh=modulation_thresh, \
                                       spacing_thresh=space_modulation_tresh, vis3d=vis3D, \
                                       d_ox=d_ox, wl=wl, na=na, pix_size=pixel_size, \
                                       apply_filt=apply_filt, filt_window=filt_window, \
                                       filt_type=filt_type, savgol_poly=savgol_poly, \
                                       option=option)

        end = time()
        # reconstruction time
        fill_reading_space(rec_time, '%.3f'%(end - start))

    else:
        # choose h_win
        if (inith_seq_select.get()):
            h_win_aux = int(optional_h_window.get())
        else:
            h_win_aux = h_win
        # separate sequences
        sequences = separate_SAIM_seq(org_path, len(angles_deg))
        previous_rec = None
        i = 0
        for sequence in sequences:
            # indicate sequence experiment number
            fill_reading_space(sq_num, f'{i + 1} / {len(sequences)}')

            # get subfolder
            subfolder = sequence[: sequence.rfind("/")]

            if (i == 0):
                h_win_aux_ = h_win_aux
                h_win_aux = h_win
                # check if there is an existing reconstruction
                try:
                    previous_rec, _ = open_images(os.path.join(subfolder, height_rec2.tif))
                except:
                    try:
                        previous_rec, _ = open_images(os.path.join(subfolder, height_rec.tif))
                    except:
                        previous_rec = None

            # copy first sequence roi to other folders in sequence
            if (i > 0):
                h_win_aux = h_win_aux_
                # copy to original sequence folder
                if (i == 1):
                    try:
                        shutil.copy(os.path.join(sequences[0][:sequences[0].rfind("/")], "height_rec_roi_definition.npy"), \
                                    os.path.join(subfolder[:subfolder[:subfolder.rfind("/")].rfind("/")], "height_rec_roi_definition.npy"))
                    except:
                        shutil.copy(os.path.join(sequences[0][:sequences[0].rfind("/")], "height_rec_polygon_roi_definition.npy"), \
                                    os.path.join(subfolder[:subfolder[:subfolder.rfind("/")].rfind("/")], "height_rec_polygon_roi_definition.npy"))
                # copy to current sequence folder
                try:
                    shutil.copy(os.path.join(sequences[0][:sequences[0].rfind("/")], "height_rec_roi_definition.npy"), \
                                os.path.join(subfolder, "height_rec_roi_definition.npy"))
                except:
                    shutil.copy(os.path.join(sequences[0][:sequences[0].rfind("/")], "height_rec_polygon_roi_definition.npy"), \
                                os.path.join(subfolder, "height_rec_polygon_roi_definition.npy"))

                # previous sequence reconstruction to use as matrix of height
                # initializations for current sequence reconstruction
                if (inith_seq_select.get()):
                    previous_rec = rec

            # look for ROI definition if present
            to_hFile = os.path.join(subfolder, f"height_rec.tif")
            # location to save reconstruction and associated files
            hImage_path = os.path.join(subfolder, to_hFile)
            try:
                roi = os.path.join(subfolder, "height_rec_roi_definition.npy")
                roi = roi.replace("//", "/")
                roi = np.load(roi, allow_pickle=True)
                roi_type = 'rect'
                roi_t.current(0)
            except:
                try:
                    roi = os.path.join(subfolder, "height_rec_polygon_roi_definition.npy")
                    roi = roi.replace("//", "/")
                    roi = np.load(roi, allow_pickle=True)
                    roi_type = 'polys'
                    roi_t.current(1)
                except:
                    roi = []

            start = time()
            SAIMimg_path, org, org_minus_bg, rec, neld_img, global_nelds_img = SAIM_reconstruction(sequence, hImage_path, \
                                           angles=angles_deg_w, angle_range=angle_range_deg_w, params=params, heights_matrix=previous_rec, \
                                           path_correction= path_correction, scipy_fit=scipy_fit, \
                                           subtract_bg=subtract_bg, roi=roi, roi_type=roi_type, \
                                           num_bg_vals=num_bg_vals, proper=proper, \
                                           keep_initial_values=keep_initial_values, use_peaks = use_peaks, \
                                           freeze_params=freeze_params, second_pass=second_pass, \
                                           power_correction=power_correction, \
                                           normalization=normalization, norm_type=norm_type, \
                                           correction=correction, interpolate_neg=interpolate_neg, \
                                           fif_metric=fif_metric, \
                                           minimize_rmse=minimize_neld, h_win=h_win_aux, h_step=h_step, \
                                           group_size=group_size, modulation_thresh=modulation_thresh, \
                                           spacing_thresh=space_modulation_tresh, vis3d=vis3D, \
                                           d_ox=d_ox, wl=wl, na=na, pix_size=pixel_size, \
                                           apply_filt=apply_filt, filt_window=filt_window, \
                                           filt_type=filt_type, savgol_poly=savgol_poly, \
                                           option=option)

            end = time()
            # reconstruction time
            fill_reading_space(rec_time, '%.3f'%(end - start))

            i += 1

        # save stacks of ROI, reconstruction in main folder
        regroup_results_time_lapse(folder, "rec_roi")
        regroup_results_time_lapse(folder, "rec2")
        regroup_results_time_lapse(folder, "rec_neg_interp2")
        regroup_results_time_lapse(folder, "rec_neld2")

        # generate and save height fluctuations image
        reconstructions, _ = open_images(os.path.join(folder, 'rec_neld2_all.tif'))
        save_tif(stack_fluctuation(reconstructions), os.path.join(folder,'height_fluctuation.tif'), 'float')
        # save fluctuation representation as an image to file
        plt.figure(figsize=(10,10))
        plt.imshow(fluct_img, 'rainbow')
        plt.axis("off")
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=20, width=3, length=10)
        cbar.ax.get_yaxis().labelpad = 40
        cbar.set_label("Fluctuation", size=30, rotation=-90)
        plt.savefig(os.path.join(folder,' height_fluctuation.png'))
        plt.savefig(os.path.join(folder,' height_fluctuation.svg'))

    fill_reading_space(h_mean, f'{np.mean(rec[~np.isnan(rec)]):.4}')
    fill_reading_space(mean_neld, f'{np.mean(neld_img[~np.isnan(neld_img)]):.4}')


def clicked_generate_lookup_tables_util(wave, dox, min_theta, max_theta, step_theta):
    '''
    Generates the height lookup tables.
    '''
    wl = int(wave.get())
    d_ox = int(dox.get())
    min_deg = float(min_theta.get())
    max_deg = float(max_theta.get())
    step_deg = float(step_theta.get()) / 2

    angles = np.arange(min_deg, max_deg + step_deg, step_deg)
    for start in angles:
        for end in angles:
            if ((end - start) > 2):
                result = read_table_heights(wl, np.round(start, 2), np.round(end, 2), d_ox=d_ox, step=0.25)
