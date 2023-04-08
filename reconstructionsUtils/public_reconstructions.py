import numpy as np
import os
import glob
from skimage import io
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import cv2
from PIL import Image
from scipy.ndimage import median_filter
from scipy.interpolate import CubicSpline
import pygpufit.gpufit as gf
import gc
from itertools import chain

from .public_interferenceEquations import *
from .public_leastSquaresLMmomentum import *
from .public_evaluationFunctions import *
from .public_filesManipulation import *
from .public_visualizationSave import *
from .public_imagesManipulation import *

# Global variables for polygon ROI sellection
# in SAIM reconstruction
# stores polygon points coordinates
pixels = []
# stores one or multiple definitions of polygones
polys = []

def remove_points(minsmaxs, data, remove):
    '''
    Removes points that are too close to each other.
    Parameters: - minsmaxs: list of indices where the mins and maxs are located.
                - data: list with data to be modified when a point is removed.
                - remove: list of indeces to remove.
    Returns: - minsmaxs: new list without indices where data was removed.
             - data: modified. When a point is removed, its closest neighbor is
                     0 or 1 by taking the opposite value of its furthest neighbor.
    '''
    if (len(remove) > 0):
        idx = remove[0]
        val1, val2 = (minsmaxs[idx], minsmaxs[idx + 1])
        minsmaxs = np.delete(minsmaxs, idx)
        data[val1] = np.abs(data[val2] - 1).astype(int)
        remove = np.delete(remove - 1, 0).astype(int)
        return remove_points(minsmaxs, data, remove)

    return minsmaxs, data

def add_points(minsmaxs, data, add):
    '''
    Add points that when two poits are too close to each other.
    Parameters: - minsmaxs: list of indices where the mins and maxs are located.
                - data: list with data to be modified when a point is removed.
                - add: list of indeces to add data.
    Returns: - minsmaxs: new list without indices where data was removed.
             - data: modified. When a point is removed, its closest neighbor is
                     0 or 1 by taking the opposite value of its furthest neighbor.
    '''
    if (len(add) > 0):
        idx = add[0]
        if (len(minsmaxs) < (idx - 5)): # & (len(minsmaxs) < (idx + 1))):
            minsmaxs_ = minsmaxs[:idx + 1]
            minsmaxs_ = np.append(minsmaxs_, (minsmaxs[idx + 1] - minsmaxs[idx]) // 2)
            minsmaxs_ = np.append(minsmaxs_, minsmaxs[idx + 1])
            minsmaxs_ = np.append(minsmaxs_, (minsmaxs[idx + 2] - minsmaxs[idx + 1]) // 2)
            minsmaxs_ = (np.append(minsmaxs_, minsmaxs[idx + 2:])).flatten()
            data[idx + 1] = np.abs(data[idx] - 1).astype(int)
            data[idx + 2] = np.abs(data[idx + 1] - 1).astype(int)
            data[idx + 3] = np.abs(data[idx + 2] - 1).astype(int)
            add = np.delete(add + 2, (0,1)).astype(int)
            return add_points(minsmaxs_, data, add)

    minsmaxs.sort()
    return minsmaxs, data

def harmonize(minsmaxs, data, to_harmonize, peaks_ids, valleys_ids):
    '''
    Next neighbors that have the same value are eliminated and their avergare postion
    is set as the new peak or valley.
    Parameters: - minsmaxs: list of indices where the mins and maxs are located.
                - data: list with data to be modified when a point is removed.
                - to_harmonize: list of indeces to remove.
                - peaks_ids: array with indices of max locations.
                - valleys_ids: array with indices of min locations.
    Returns: - minsmaxs: new list without indices where data was removed and new indices
                         where data was added.
             - data: modified. When two neighbors are removed, at their average position a
                     point is added with their value.
    '''
    if (len(to_harmonize) > 0):
        idx = to_harmonize[0]
        val1, val2 = (minsmaxs[idx], minsmaxs[idx + 1])
        # when points are peaks, choose the tallest peak
        if (val1 in peaks_ids):
            if (data[val1] > data[val2]):
                minsmaxs = np.delete(minsmaxs, idx + 1)
            else:
                minsmaxs = np.delete(minsmaxs, idx)

        # when points are valleys, choose the lowest valley
        else:
            if (data[val1] < data[val2]):
                minsmaxs = np.delete(minsmaxs, idx + 1)
            else:
                minsmaxs = np.delete(minsmaxs, idx)

        #minsmaxs = np.delete(minsmaxs, idx)
        # minsmaxs[idx] = (val1 + val2) // 2
        # data[minsmaxs[idx]] = data[val1]
        to_harmonize = np.delete(to_harmonize - 1, 0).astype(int)
        return harmonize(minsmaxs, data, to_harmonize, peaks_ids, valleys_ids)

    return minsmaxs, data

def readjust_fit_window(remove, valleys, angles, start_idx, end_idx):
    # if more than one peak to remove, consider only the first and last ones
    if (len(remove) > 1):
        # if the points are significantly appart
        if ((remove[-1] - remove[0]) > 20):
            start_idx = valleys[valleys > remove[0]][0] - 3
            end_idx = valleys[valleys > remove[-1]][0] + 3

        # if peaks are too close take only the first peak
        else:
            remove = [remove[0]]

    # if only one peak to remove, determine if is closest to the end or beginning
    elif (len(remove) == 1):
        # if closest to the beginning, keep data to the right of the peak
        if (np.abs(angles[remove[0]] - angles[0]) < np.abs(angles[remove[0]] - angles[-1])):
            start_idx = valleys[valleys > remove[0]][0] - 3
            #end_idx = len(dat_) - 1

        # if closest to the end, keep data to the left of the peak
        else:
            #start_idx = 0
            end_idx = valleys[valleys < remove[0]][-1] + 3

    return start_idx, end_idx

def normalize_by_modulations(minsmaxs, data):
    '''
    This function normalize each modulation in the data separately on the range
    of indexes given by minsmaxs.
    Parameters: - minsmaxs: organized array with location  of mins and maxs in data.
                - data: array, data to normalize.
    Returns: array with normalized data on the range give by minsmaxs.
    '''
    new_data = np.zeros_like(data)
    for start, end in zip(minsmaxs[:-1], minsmaxs[1:]):
        modulation = data[start: end + 1]
        new_data[start: end + 1] = (modulation - min(modulation)) / (max(modulation) - min(modulation))

    return new_data

def good_extrema(data, angles, minsmaxs, group_size=3, thresh=0.1, second_pass=False, spacing_thresh=0.4, verbose=0):
    '''
    Changes the values of minmaxs to include only
    "nice" modulations for the fit. "Nice" modulations
    are those with an amplitude bigger than thresh and
    are grouped by 'group_size' or more modulations.

    Parameters: - data: array.
                - angles: array.
                - minsmaxs: index locations of mins and maxs in data.
                - group_size: int, minimum number of modulations to form a group.
                - thresh: float to evaluate amplitude of modulations.
                - second_pass: boolean, if True, the distance between neigboring
                               extrema from left to right is also considered.
                - spacing_thresh: float to evaluate width of modulations.
    Returns: modified minsmaxs to include only 2 or more consecutive
             modulations with an amplitude higher than thresh.
             An empty list otherwise.
    '''
    # given that the data is normalized, for the threshold
    # we assume that the maximum amplitude is 1.

    thresh = thresh * np.mean(np.abs(data[minsmaxs[1:]] - data[minsmaxs[:-1]]))

    if (second_pass):
        spacing_mean = np.mean(np.abs(angles[minsmaxs[1:]] - angles[minsmaxs[:-1]]))

    if (verbose):
        print(f"Average amplitude: {np.mean(np.abs(data[minsmaxs[1:]] - data[minsmaxs[:-1]]))}")
        print(f"Average spacing: {np.mean(angles[minsmaxs[1:]] - angles[minsmaxs[:-1]])}")

    # separate individual modulations
    # a modulation is three neighbors of minsmaxs
    # triplets is a tuple with three neighbors, their relative amplitude
    # and a boolean telling if their amplitudes are higher than thresh
    triplets = []
    for i in range(len(minsmaxs) - 2):
        tri_data = data[minsmaxs[i: i + 3]]
        tri_diff = np.abs(tri_data[1:] - tri_data[:-1])
        if (second_pass == False):
            triplets.append(((i, i + 1, i + 2), tri_diff, np.any(tri_diff < thresh)))
        else:
            tri_spacing_diff = np.abs(angles[minsmaxs[i + 1: i + 3]] - angles[minsmaxs[i: i + 2]])
            triplets.append(((i, i + 1, i + 2), tri_diff, np.any((tri_diff < thresh) | ((tri_spacing_diff < spacing_mean * (1 - spacing_thresh)) | (tri_spacing_diff > spacing_mean * (1 + spacing_thresh))))))

    # groups is a collection of neighboring triplets
    group = []
    # a collection of groups
    groups = []
    # array with the size of each group
    len_groups = []
    for triple in triplets:
        # consider triplets with the right amplitude
        if (triple[2] == False):
            # when empty group
            if (len(group) == 0):
                group.append(triple)
            # append to group only if triple is a neighbour
            elif (group[-1][0][1:] == triple[0][:2]):
                group.append(triple)
            # if not a neighbor, create new group
            else:
                groups.append(group)
                len_groups.append(len(group))
                group = [triple]

    groups.append(group)
    len_groups.append(len(group))

    # # idx of biggest group
    # idx = np.argmax(len_groups)
    #
    # # if more than 3 consecutive modulations
    # if (len_groups[idx] >= 7):
    #     # redefine minsmaxs
    #     return minsmaxs[list(range(groups[idx][0][0][0], groups[idx][-1][0][-1] + 1))]

    # idxs with groups with more than the specified modulations
    idxs = np.argwhere(np.array(len_groups) >= (group_size * 2 + 1)).flatten()

    # return groups with 3 consecutive modulations or more
    if (len(idxs) > 0):
        # list of redefined minsmaxs
        return [minsmaxs[list(range(groups[idx][0][0][0], groups[idx][-1][0][-1] + 1))] for idx in idxs]

    else:
        return []

def reassign_values_close_extrema(minsmaxs, data_org, thresh=0.1):
    '''
    For extrema neighbors who have an ampltidue difference less than
    thresh, it reassigns the mean value between the next and previous extrema.
    '''
    data = np.copy(data_org)
    thresh = thresh * np.mean(np.abs(data_org[minsmaxs[1:]] - data_org[minsmaxs[:-1]]))
    for i in range(len(minsmaxs) - 1):
        if (np.abs(data_org[minsmaxs[i]] - data_org[minsmaxs[i + 1]]) < thresh):
            data[minsmaxs[i]] = (data_org[minsmaxs[i] + 1] + data_org[minsmaxs[i] - 1]) / 2

    return data

def add_subfigure(angles, dat, n, ax, color, label=r"$I_k(\theta)$", linestyle='-', marker='o', linewidth=3, size=20, markersize=7):
    font = "Arial"
    correction = 5
    pad = 10
    width = 2
    length = 14
    width_minor = 2
    length_minor = 7

    # angles in air
    angles_air = np.around(np.arcsin(np.sin(np.array(angles) * np.pi / 180) * n / 1) * 180 / np.pi, 2)

    # define axes
    ax.set_xlim((np.min(angles_air), np.max(angles_air)))
    ax.set_xlabel(r'$\theta_{\rm w}$ (°)', fontname=font, size=size)
    ax.set_ylabel(r"$I_k$ (a.u.)", fontname=font, size=size)
    ax.tick_params(axis='both', labelsize=size-correction, width=width, length=length)

    # axis for angles in air
    axb = ax.twiny()
    #_ = axb.plot(angles_air, np.zeros(len(angles_air)))
    axb.set_xlim((np.min(angles_air), np.max(angles_air)))
    axb.set_xlabel(r'$\theta_{\rm air}$ (°)', fontname=font, labelpad=pad, size=size)
    axb.tick_params(axis='x', labelsize=size-correction, width=width, length=length)

    # plot normalized raw data
    ax.plot(angles_air, dat, linestyle=linestyle, linewidth=linewidth, marker=marker, markersize=markersize, label=label, color=color)
    ax.set_ylim(-0.014, 1.014)
    ax.tick_params(which='minor', width=width_minor, length=length_minor)
    ax.minorticks_on()
    # ax.tick_params(axis='x',which='minor',bottom=True)
    # ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    axb.tick_params(which='minor', width=width_minor, length=length_minor)
    axb.minorticks_on()
    # axb.tick_params(axis='x',which='minor',bottom=True)
    # axb.xaxis.set_minor_locator(AutoMinorLocator(5))

    return None

def make_space_above(axes, topmargin=1):
    '''
    Increase figure size to make topmargin (in inches) space for
    titles, without changing the axes sizes.
    From https://stackoverflow.com/a/55768955
    '''
    fig = axes.flatten()[0].figure
    s = fig.subplotpars
    w, h = fig.get_size_inches()

    figh = h - (1-s.top)*h  + topmargin
    fig.subplots_adjust(bottom=s.bottom*h/figh, top=1-topmargin/figh)
    fig.set_figheight(figh)

def show_figure(ax1):
    font_props = font_manager.FontProperties(family='Arial',
                                       style='normal', size=15)
    ax1.set_ylim(0, 1)
    ax1.legend(loc=(1.01, 0.5), prop=font_props, frameon=False)
    plt.show()

def SAIM_one_pixel(angles, data, angle_range=None, prominence_factor=0, \
                    scipy_fit=False, use_gpu=False, \
                    normalization=True, norm_type='minmax',\
                    keep_initial_values=False, use_peaks=False, \
                    freeze_params=False, extend_minsmaxs=False, points_to_remove=0, \
                    group_size=3, modulation_thresh=0.2,
                    second_pass=False, spacing_thresh=0.4, \
                    h_init=None, h_win=500, h_step=100, \
                    d_ox=1000, n=1.3355, n_si=4.3707, n_ox=1.4630, wl=488, \
                    apply_filt=False, filt_window=3, filt_type='median', savgol_poly=3, \
                    reg_val=0.1, beta1=1/3, beta2=2, max_iterations=100, exit_val=2, fletcher=True, \
                    add_momentum=True, alpha=0.1, title_str="", path='', verbose=0, use_tk=False, tk_frame=None):
    '''
    SAIM one pixel
    Parameters: - angles: in degrees, list of angles in the imaging medium (water) at which data was taken.
                - extend_minsmaxs: boolean, if True don't use as end points a max or a min,
                                   instead go 2 pixels before at the beginning and 2 ahead at the end.
                                   Default False.
                - points_to_remove: points to remove at the begiinning and end of list of mins and maxs.
                                    Default 0.
                - group_size: int, minimum number of modulations to form a group.
                - modulation_thresh: float, relative modulation with respect to mean peak to peak amplitude.
                                     Peaks with amplitude bigger than that are considered and the rest ignored.
                - second_pass: boolean, indicates if this is a second pass for modulation assignment.
                - spacing_thresh: float, used when second_pass is True.
                - path: str, location to save figures.
                - use_tk: indicate if plots are going to be rendered using tkinter.
                - tk_frame: tkinter canvas to place plot.
    '''
    # select data from angles indicated by the user
    if (np.all(angle_range)):
        # find indexes corresponding to angles
        idx_init = np.argmin(np.abs(angles - angle_range[0]))
        idx_end = np.argmin(np.abs(angles - angle_range[1])) + 1
        # select correct angles
        angles = angles[idx_init: idx_end]
        # select correct data
        data = data[idx_init: idx_end]

    # normalize
    data = (data - np.min(data)) / (np.max(data) - np.min(data))

    # shift data along the intensity axis
    #dat = data - (np.min(data) - 1)
    dat = np.copy(data)

    # smooth data
    if (apply_filt):
        if (filt_type == 'median'):
            dat = median_filter(dat, filt_window)

        elif (filt_type == 'savgol'):
            dat = savgol_filter(dat, filt_window, savgol_poly)

        #smooth_data = np.copy(data)

    # normalize
    dat = (dat - np.min(dat)) / (np.max(dat) - np.min(dat))

    # resample data by taking average value between data points
    dat = [dat[i // 2] if (i % 2 == 0) else ((dat[i // 2] + dat[i // 2 + 1]) / 2) for i in range(len(data) * 2 - 1)]
    dat = np.array(dat)

    # redefine list of angles
    angles_re = [angles[i // 2] if (i % 2 == 0) else ((angles[i // 2] + angles[i // 2 + 1]) / 2) for i in range(len(angles) * 2 - 1)]
    angles_re = np.array(angles_re)
    # angles in air
    #angle_range_air = (np.arcsin(np.sin(np.array(angle_range) * np.pi / 180) * n / 1) * 180 / np.pi).astype('int')
    angles_air = np.around(np.arcsin(np.sin(np.array(angles_re) * np.pi / 180) * n / 1) * 180 / np.pi, 2)

    # plot processed signal
    # if (verbose):
    #     ax1.plot(angles_re, dat, linestyle='--', marker='o', label="smoothed-resampled data")
    #     show_figure(ax1)
    #     ax1 = define_figure(angles, data, n)

    # file containing theoretical number of peaks
    # as a function of height in ascending order for height
    # if (df is None):
    #     df = read_table_heights(wl, int(angles_air[0]), int(angles_air[-1]), d_ox=d_ox, step=0.25)

    # try different distances
    prominence_ = np.std(dat)
    distances = [2] # minimum separation between peaks or valleys
    distance_fit_test = {}
    for distance in distances:
        # create figure and plot raw data
        if (verbose):
            # define figure
            # definitons for plot
            fit_color = 'tab:orange'
            raw_color = 'gray'
            alph = 1
            font = "Arial"
            size = 35
            markersize = 9
            raw_label = r"$I_k(\theta)$"
            loc = (1.05,0.0)
            vlines_color = (1, 0, 0)
            vline_alpha = 0.8
            vline_ymin = -0.014
            vline_ymax = 1.014
            vlines_thickness = 6
            line_width = 4

            plt.rcParams.update({'font.family':'arial'})
            font_props = font_manager.FontProperties(family='Arial', style='normal', size=size)
            # create figure handler
            fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(32,30), squeeze=False)
            if (use_tk):
                canvas = FigureCanvasTkAgg(fig, tk_frame)
            # definitons for plot
            add_subfigure(angles, data, n, ax1, raw_color, label=raw_label, linewidth=line_width, size=size, markersize=markersize)

        # try different prominences
        params_fits_p = []
        params_inits_p = []
        rmses_p = []
        r2s_p = []
        peakLocDiffs_p = []
        gnelds_p = []
        params_inits_p = []
        used_angles = []
        prominence_factors =  [0.01] # np.arange(0.01, 0.051, 0.001) #
        i = 0
        for prominence_factor in prominence_factors:
            # peaks in resmapled data
            prominence = prominence_ * prominence_factor
            peaks_ids = find_peaks(dat, distance=distance, prominence=prominence)[0]
            # valleys in resampled data
            valleys_ids = find_peaks(np.max(dat) - dat, distance=distance, prominence=prominence)[0]

            # create data where maxs are set to global max and mins to global min
            inter = np.copy(dat)
            inter[peaks_ids] = np.max(dat)
            inter[valleys_ids] = np.min(dat)
            # inter = inter[minsmaxs[evaluation[0]]: minsmaxs[evaluation[-1] + 1] + 1]

            # plot found peaks
            if (verbose):
                max_color = (0.93, 0.35, 0)
                min_color = (0.12, 0.38, 0.57) #(0.74, 0.6, 0.48)
                marker_size_maxmin = 15
                smooth_size = 4
                smooth_color = "tab:pink"
                smooth_line = "--"
                smooth_alpha = 1
                smooth_zorder = 0 # tp draw it first
                smoot_marker_size = 10

                ax1.plot(angles_air[peaks_ids], dat[peaks_ids], label=r"$\theta_{\max}$", color=max_color, linestyle="", marker="o", markerfacecolor=max_color, markersize=marker_size_maxmin)
                ax1.plot(angles_air[valleys_ids], dat[valleys_ids], label=r"$\theta_{\min}$", color=min_color, linestyle="", marker="o", markerfacecolor=min_color, markersize=marker_size_maxmin)
                ax1.legend(fontsize=size, loc=loc, prop=font_props)

                add_subfigure(angles, data, n, ax2, raw_color, label=raw_label, linewidth=line_width, size=size, markersize=markersize)
                if (apply_filt):
                    ax2.plot(angles_air, dat, label=r"$I_{\rm i\_med}$", color=smooth_color, alpha=smooth_alpha, linestyle="--", linewidth=smooth_size, marker="o", zorder=smooth_zorder, markersize=smoot_marker_size)
                else:
                    ax2.plot(angles_air, dat, label=r"$I_{\rm i\_med}$", color=smooth_color, alpha=smooth_alpha, linestyle=smooth_line, linewidth=smooth_size, marker="o", zorder=smooth_zorder, markersize=smoot_marker_size)
                #ax2.plot(angles_re[minsmaxs], dat[minsmaxs], label="found mins-maxs", color='tab:orange', linestyle=" ", marker="o", markersize=10)
                ax2.plot(angles_air[peaks_ids], dat[peaks_ids], label=r"$\theta_{\max}$", color=max_color, linestyle=" ", marker="o", markerfacecolor=max_color, markersize=marker_size_maxmin)
                ax2.plot(angles_air[valleys_ids], dat[valleys_ids], label=r"$\theta_{\min}$", color=min_color, linestyle=" ", marker="o", markerfacecolor=min_color, markersize=marker_size_maxmin)
                ax2.legend(loc=loc, prop=font_props)

            # # join maxs and mins plus frontier values
            # minsmaxs_ = np.concatenate([peaks_ids, valleys_ids])
            # minsmaxs_.sort()

            # look for extrema in original normalized data
            peak_org_id = find_peaks(data, distance=distance, prominence=np.std(data) * prominence_factor)[0]
            valley_org_id = find_peaks(np.max(data) - data, distance=distance, prominence=np.std(data) * prominence_factor)[0]
            minsmaxs_org = np.concatenate([peak_org_id, valley_org_id])
            minsmaxs_org.sort()

            # reassign values for extrema that are too close
            # (choose mean between next and previous extrema)
            dat = reassign_values_close_extrema(minsmaxs_org, data, thresh=0.1)
            # resample dat by taking average value between data points
            dat = [dat[i // 2] if (i % 2 == 0) else ((dat[i // 2] + dat[i // 2 + 1]) / 2) for i in range(len(data) * 2 - 1)]
            dat = np.array(dat)
            # peaks in new dat
            peaks_ids = find_peaks(dat, distance=distance, prominence=np.std(dat) * prominence_factor)[0]
            # valleys in new dat
            valleys_ids = find_peaks(np.max(dat) - dat, distance=distance, prominence=np.std(dat) * prominence_factor)[0]
            inter = np.copy(dat)
            inter[peaks_ids] = np.max(dat)
            inter[valleys_ids] = np.min(dat)
            minsmaxs_ = np.concatenate([peaks_ids, valleys_ids])
            minsmaxs_.sort()

            # # take only minsmaxs with neighbors at both sides
            # minsmaxs = minsmaxs[2:-2]
            #
            # remove points that are too close in amplitude
            # mmdiff = np.abs(dat[minsmaxs[1:]] - dat[minsmaxs[:-1]])
            # mmavg = np.mean(mmdiff)
            # factor = 0.3
            # to_remove = (np.argwhere(mmdiff < mmavg * factor)).flatten()
            # minsmaxs, inter = remove_points(minsmaxs, inter, to_remove)
            #
            # # add points when two points too far from each other
            # # mmdiff = minsmaxs[1:] - minsmaxs[:-1]
            # # mmavg = np.mean(mmdiff)
            # factor = 1.7
            # to_add = (np.argwhere(mmdiff > mmavg * factor)).flatten()
            # minsmaxs, inter = add_points(minsmaxs, inter, to_add)
            #
            # harmonize minsmaxs (if two close neighbors are the same value,
            # then eliminate them and take their average position as new index)
            # mmidxdiff = inter[minsmaxs[1:]] - inter[minsmaxs[:-1]]
            # to_harmonize = (np.argwhere(mmidxdiff == 0)).flatten()
            # minsmaxs, inter = harmonize(minsmaxs, inter, to_harmonize, peaks_ids, valleys_ids)
            #
            # # redefine inter with new minsmaxs
            # #inter = inter[minsmaxs[0]: minsmaxs[-1] + 1]
            # # take data 3 points before and after minsmaxs limits
            # start_idx = minsmaxs[0] - 3
            # if (start_idx < 0):
            #     start_idx = 2
            #
            # end_idx = minsmaxs[-1] + 3
            # if (end_idx > (len(angles_re) - 1)):
            #     end_idx = len(angles_re) - 2
            #
            # # # calculate distances between minsmaxs
            # # mmdiff = minsmaxs[1:] - minsmaxs[:-1]
            # # mmavg = np.mean(mmdiff)
            # # # criteria to add delete points
            # # add_crit = mmavg * 1.5
            # # del_crit = mmavg * 0.4
            # # # find positions to add/remove points
            # # add_at_idxs = np.argwhere(mmdiff > add_crit)
            # # del_at_idxs = np.argwhere(mmdiff < del_crit)
            #
            # angles_re_ = angles_re[start_idx: end_idx + 1]
            # angles_air_ = angles_air[start_idx: end_idx + 1]
            # minsmaxs = np.concatenate([[start_idx], minsmaxs, [end_idx]])

            # get different windows of good modulations
            # (a good modulation is at leats 3 consecutive peaks)
            # modulation threshold
            minsmaxsses_dict_aux = {}
            for mod_thresh in np.arange(0.05, 0.55, 0.05).round(2):
                minsmaxsses_dict_aux[(mod_thresh, 'none')] = list(map(tuple, good_extrema(dat, angles_air, minsmaxs_, group_size=group_size, thresh=mod_thresh, second_pass=False, spacing_thresh=spacing_thresh, verbose=0)))
                for sp_thresh in np.arange(0.5, 1.05, 0.05).round(2):
                    minsmaxsses_dict_aux[(mod_thresh, sp_thresh)] = list(map(tuple, good_extrema(dat, angles_air, minsmaxs_, group_size=group_size, thresh=mod_thresh, second_pass=True, spacing_thresh=sp_thresh, verbose=0)))

            for (mod_thresh, sp_thresh), groups in minsmaxsses_dict_aux.items():
                if (sp_thresh != 'none'):
                    minsmaxsses_dict_aux[(mod_thresh, sp_thresh)].extend(list(chain.from_iterable([good_extrema(dat, angles_air, np.array(group), group_size=group_size, thresh=mod_thresh, second_pass=True, spacing_thresh=sp_thresh, verbose=0) for group in groups for sp_thresh in np.arange(0.5, 1.05, 0.05) for mod_thresh in np.arange(0.15, 0.40, 0.05)])))
                    minsmaxsses_dict_aux[(mod_thresh, sp_thresh)] = list(set(map(tuple, minsmaxsses_dict_aux[(mod_thresh, sp_thresh)])))

            minsmaxsses_dict = {}
            for parameters, groups in minsmaxsses_dict_aux.items():
                for group in groups:
                    minsmaxsses_dict[group] = minsmaxsses_dict.get(group, [])
                    minsmaxsses_dict[group].append(parameters)

            # minsmaxsses = [good_extrema(dat, angles_air, minsmaxs_, group_size=group_size, thresh=mod_thresh, second_pass=second_pass, spacing_thresh=sp_thresh, verbose=verbose) for sp_thresh in np.arange(0.5, 1.05, 0.05) for mod_thresh in np.arange(0.15, 0.40, 0.05)]
            # minsmaxsses.extend([good_extrema(dat, angles_air, minsmaxs_, group_size=group_size, thresh=mod_thresh, second_pass=False, spacing_thresh=spacing_thresh, verbose=verbose) for mod_thresh in np.arange(0.15, 0.40, 0.05)])
            # # convert everything to one list (unnest nested groups)
            # minsmaxsses = list(chain.from_iterable(minsmaxsses))
            # # remove repeated groups, the result is a list of tuples
            # minsmaxsses = list(set(map(tuple, minsmaxsses)))
            # #minsmaxsses = good_extrema(dat, angles_air, minsmaxs_, group_size=group_size, thresh=modulation_thresh, second_pass=second_pass, spacing_thresh=spacing_thresh, verbose=verbose)

            #if (len(minsmaxsses) == 0):
            if (len(minsmaxsses_dict) == 0):
                params_fits_p.append(np.array([-1,-1,-1]))
                params_inits_p.append(np.array([-1,-1,-1]))
                rmses_p.append(-1)
                r2s_p.append(-1)
                peakLocDiffs_p.append(-1)
                gnelds_p.append(-1)
                params_inits_p.append(-1)
                angles_re_ = angles_re
                angles_air_ = angles_air
                dat_ = dat
                used_angles.append(angles_re)

                continue

            # try the fit for different groups of good modulations
            groups_fit_test = {}
            #for minsmaxs in minsmaxsses:
            for minsmaxs, parameters in minsmaxsses_dict.items():
                minsmaxs = np.array(minsmaxs)
                # consider 4 cases:
                # - all the peaks and valleys in the group
                # - if possible, remove only last peak or valley
                # - if possible, remove only first peak or valley
                # - if possible, remove first and last peaks or valleys
                #minsmaxs_to_try = [minsmaxs, minsmaxs[:-points_to_remove], minsmaxs[points_to_remove:], minsmaxs[points_to_remove:-points_to_remove]]
                if (points_to_remove == 0):
                    minsmaxs_to_try = [minsmaxs[points_to_remove:]]

                else:
                    minsmaxs_to_try = [minsmaxs[points_to_remove:], minsmaxs[points_to_remove:-points_to_remove]]

                if (extend_minsmaxs):
                    minsmaxsendss = []
                    for mms in minsmaxs_to_try:
                        if (len(mms) < 7):
                            continue
                        # take data 3 points before and after minsmaxs limits
                        start_idx = mms[0] - 3
                        if (start_idx < 0):
                            start_idx = 1

                        end_idx = mms[-1] + 3
                        if (end_idx > (len(angles_re) - 1)):
                            end_idx = len(angles_re) - 1

                        if ((start_idx != mms[0]) & (end_idx != mms[-1])):
                            minsmaxsendss.append(np.concatenate([[start_idx], mms, [end_idx]]))

                        elif (start_idx == mms[0]):
                            minsmaxsendss.append(np.concatenate([mms, [end_idx]]))

                        elif (end_idx == mms[-1]):
                            minsmaxsendss.append(np.concatenate([[start_idx], mms]))

                else:
                    minsmaxsendss = minsmaxs_to_try

                # plot used peaks
                if (verbose):
                    add_subfigure(angles, data, n, ax3, raw_color, label=raw_label, linewidth=line_width, size=size, markersize=markersize)
                    if (apply_filt):
                        ax3.plot(angles_air, dat, label=r"$I_{\rm i\_med}$", color=smooth_color, alpha=smooth_alpha, linestyle="--", linewidth=smooth_size, marker="o", zorder=smooth_zorder, markersize=smoot_marker_size)
                    else:
                        ax3.plot(angles_air, dat, label=r"$I_{\rm i\_med}$", color=smooth_color, alpha=smooth_alpha, linestyle=smooth_line, linewidth=smooth_size, marker="o", zorder=smooth_zorder, markersize=smoot_marker_size)
                    ax3.plot(angles_air[minsmaxs_], dat[minsmaxs_], label="non-selected\n"+r"$\theta_{\min-\max}$", color='tab:purple', linestyle=" ", marker="o", markersize=marker_size_maxmin)
                    ax3.plot(angles_air[minsmaxsendss[0][1:-1]], dat[minsmaxsendss[0]][1:-1], label="selected\n"+r"$\theta_{\min-\max}$", color='g', linestyle=" ", marker="o", markersize=marker_size_maxmin)
                    ax3.plot(angles_air[minsmaxsendss[0][1:-1]], inter[minsmaxsendss[0][1:-1]], linestyle=" ", marker="o", color='k')
                    # plot vertical lines are angle limits
                    ax3.vlines([angles_air[minsmaxsendss[0][1]-3], angles_air[minsmaxsendss[0][-2]+3]], ymin=vline_ymin, ymax=vline_ymax, color=vlines_color, alpha=vline_alpha, linestyles="-", linewidth=vlines_thickness)#, label=r"$\theta_{\rm fit-lim}$")
                    if (i == 0):
                        ax3.legend(fontsize=size, loc=loc, prop=font_props)
                        i += 1

                # test the fit for all the possible cases of mixsmaxsends
                fit_tests_results = {}
                for minsmaxsends in minsmaxsendss:

                    # normalize data between modulations
                    dat_ = normalize_by_modulations(minsmaxsends, dat)

                    # define start and end indeces
                    start_idx = minsmaxsends[1] - 3
                    end_idx =  minsmaxsends[-2] + 3

                    # adjust angles to fitting window
                    angles_re_ = angles_re[start_idx: end_idx + 1]
                    angles_air_ = angles_air[start_idx: end_idx + 1]

                    dat_ = dat_[start_idx: end_idx + 1]

                    # # cubic spline
                    # smooth_cs = CubicSpline(angles_re[minsmaxsends], inter[minsmaxsends], bc_type='clamped')
                    # dat_ = smooth_cs(angles_re_)
                    # #dat_ = inter[start_idx: end_idx + 1]

                    # if global min negative, shift data
                    if (np.min(dat_) < 0):
                        dat_ = dat_ - np.min(dat_)

                    # renormalize
                    dat_ = (dat_ - np.min(dat_)) / (np.max(dat_) - np.min(dat_))

                    # plot initial spline data
                    if (verbose):
                        dat1 = dat[start_idx: end_idx + 1]
                        pld_current = peakloc_diff(dat1, dat_, angles_air_, verbose=verbose)
                        add_subfigure(angles, data, n, ax4, raw_color, label=raw_label, linewidth=line_width, size=size, markersize=markersize)
                        ax4.plot(angles_air_, dat_,label=r"$I_{\rm rs}$", \
                                 color="tab:blue", linestyle="--", marker="o", linewidth=line_width, markersize=markersize)
                        ax4.vlines([angles_air_[0], angles_air_[-1]], ymin=vline_ymin, ymax=vline_ymax, color=vlines_color, alpha=vline_alpha, linestyles="-", linewidth=vlines_thickness)#, label=r"$\theta_{\rm fit-lim}$")

                    if (h_init == None):
                        # initialization with respect to peaks
                        df = read_table_heights(wl, np.round(angles_air_[0], 2), np.round(angles_air_[-1], 2), d_ox=d_ox, step=0.25)
                        max_heights = list(df.height_nm[df.num_peaks == df.num_peaks.iloc[np.argmin(np.abs(df.num_peaks - len(dat_[dat_ == 1])))]])
                        min_heights = list(df.height_nm[df.num_valleys == df.num_valleys.iloc[np.argmin(np.abs(df.num_valleys - len(dat_[dat_ == 0])))]])
                        min_min_heights = min([min_heights[0], max_heights[0]])
                        max_max_heights = max([min_heights[-1], max_heights[-1]])
                        h0 = (min_min_heights + max_max_heights) / 2
                        h_win = max_max_heights - min_min_heights

                        # free memory used by dataframe
                        del df
                        gc.collect()

                    else:
                        h0 = h_init

                    # inspect height range
                    #rmses = []
                    #r2s = []
                    peakLocDiffs = []
                    gnelds = []
                    params_fits = []
                    params_inits = []

                    # with GPU if present on the system
                    if (use_gpu):
                        #continue
                        # choose gpufit model
                        if (wl == 488):
                            model = gf.ModelID.SAIM_488
                        elif (wl == 560):
                            model = gf.ModelID.SAIM_560
                        else:
                            model = gf.ModelID.SAIM_647

                        # array of parameters to try
                        params_inits = np.array([[1, 0, h] for h in np.arange(h0 - h_win / 2, h0 + h_win / 2 + h_step, h_step)], dtype='float32')

                        # run Gpufit
                        params_fits, _, chi_squares, number_iterations, execution_time = gf.fit(np.array([dat_] * len(params_inits), dtype='float32'), None, \
                                                                                                    model, params_inits, \
                                                                                                    #tolerance = 1e-10, \
                                                                                                    #max_number_iterations=1000, \
                                                                                                    user_info=(angles_re_ * np.pi / 180).astype('float32') \
                                                                                                   )

                        for params_found in params_fits:
                            if (verbose):
                                print(f"h = {params_found[2]} nm")
                            peakLocDiffs.append(peakloc_diff(dat_, intensity(angles_re_ * np.pi / 180, *params_found, n, n_ox, n_si, wl, d_ox), angles_air_, verbose=verbose))
                            gnelds.append(peakloc_diff(dat, intensity(angles_re * np.pi / 180, *params_found, n, n_ox, n_si, wl, d_ox), angles_air, verbose=verbose))

                    # if no GPU, use CPU
                    else:
                        for h in np.arange(h0 - h_win / 2, h0 + h_win / 2 + h_step, h_step):
                            params_init = np.array([1/3, 0, h])
                            params_found = LM_with_momentum(angles_re_, dat_, n, n_ox, n_si, wl, d_ox, \
                                                      scipy_fit=scipy_fit, \
                                                      params=params_init, normalization=False, \
                                                      norm_type =norm_type, \
                                                      proper=proper, keep_initial_values=keep_initial_values, \
                                                      use_peaks=use_peaks, freeze_params=freeze_params, \
                                                      apply_filt=False, filt_window=filt_window, \
                                                      savgol_poly=savgol_poly \
                                                     )

                            if (verbose):
                                print(f"h = {params_found[2]} nm")
                            peakLocDiffs.append(peakloc_diff(dat_, intensity(angles_re_ * np.pi / 180, *params_found, n, n_ox, n_si, wl, d_ox), angles_air_, verbose=verbose))
                            gnelds.append(peakloc_diff(dat, intensity(angles_re * np.pi / 180, *params_found, n, n_ox, n_si, wl, d_ox), angles_air, verbose=verbose))
                            params_inits.append(params_init)
                            params_fits.append(params_found)

                    peakLocDiffs = np.array(peakLocDiffs)
                    gnelds = np.array(gnelds)
                    peakLocDiffs[(peakLocDiffs <= 0) & (~np.isnan(peakLocDiffs))] = np.max(peakLocDiffs)
                    min_idx = np.argmin(peakLocDiffs)


                    # check peaks for fit data and spline
                    fit_data = intensity(angles_re_ * np.pi / 180, *params_fits[min_idx], n, n_ox, n_si, wl, d_ox)
                    if ((len(find_peaks(fit_data)[0]) == len(find_peaks(dat_)[0])) \
                        | (len(find_peaks(max(fit_data) - fit_data)[0]) == len(find_peaks(max(dat_) - dat_)[0])) \
                       ):
                        fit_tests_results[peakLocDiffs[min_idx]] = (params_fits[min_idx], \
                                                                    params_inits[min_idx], \
                                                                    rmse(dat_, fit_data), \
                                                                    r2(dat_, fit_data), \
                                                                    peakLocDiffs[min_idx], \
                                                                    gnelds[min_idx], \
                                                                    params_inits[min_idx], \
                                                                    angles_air_ \
                                                                   )

                    else:
                        fit_tests_results[peakLocDiffs[min_idx]] = (np.array([-1,-1,-1]), \
                                                                    np.array([-1,-1,-1]), \
                                                                    -1, \
                                                                    -1, \
                                                                    -1, \
                                                                    -1, \
                                                                    -1, \
                                                                    angles_air_ \
                                                                   )


                    if (verbose):
                        dat1 = dat[start_idx: end_idx + 1]
                        pld_current = peakLocDiffs[min_idx]
                        gnedl_current = gnelds[min_idx]
                        add_subfigure(angles, data, n, ax5, raw_color, label=raw_label, linewidth=line_width, size=size, markersize=markersize)
                        ax5.plot(angles_air_, dat_,label=r"$I_{\rm rs}$", \
                                 color="tab:blue", linestyle="--", marker="o", linewidth=line_width, markersize=markersize)
                        ax5.plot(angles_air_, fit_data, label=r"$I_{\rm fit}$", \
                                 color=fit_color, linestyle="--", marker="o", linewidth=line_width, markersize=markersize)
                        # plot vertical lines
                        ax5.vlines([angles_air_[0], angles_air_[-1]], ymin=vline_ymin, ymax=vline_ymax, color=vlines_color, alpha=vline_alpha, linestyles="-",  linewidth=vlines_thickness)#, label=r"$\theta_{\rm fit-lim}$")

                    if (verbose):
                        pld_current = peakLocDiffs[min_idx]
                        gneld_current = gnelds[min_idx]
                        figb, ax = plt.subplots(1,1, figsize=(10,10))
                        add_subfigure(angles, data, n, ax, raw_color, label=raw_label, linewidth=line_width, size=size, markersize=markersize)
                        ax.plot(angles_air_, dat[start_idx: end_idx + 1], label=r"$I_{\rm i\_med}$", color=smooth_color, alpha=smooth_alpha, linestyle='-', linewidth=smooth_size, marker="o", zorder=smooth_zorder, markersize=smoot_marker_size)
                        ax.plot(angles_air_, dat_, label=r"$I_{\rm rs}$"+f"\nNELD={float(pld_current):.4}", linestyle="--", marker="o")
                        ax.plot(angles_air_, fit_data, label=r"best $I_{\rm fit}$"+f"\nNELD={float(pld_current):.4}\nh={params_fits[min_idx][-1]:.6}nm\ngNELD={float(gneld_current):.4}", color=fit_color, linestyle="--", marker="o")
                        # plot vertical lines
                        ax.vlines([angles_air_[0], angles_air_[-1]], ymin=vline_ymin, ymax=vline_ymax, color=vlines_color, alpha=vline_alpha, linestyles="-", linewidth=vlines_thickness)#, label=r"$\theta_{\rm fit-lim}$")
                        ax.legend(fontsize=size, loc=loc, prop=font_props)
                        figb.savefig(path + f"NELD={float(pld_current):.4}.svg")
                        figb.show()

                min_plds = np.array(list(fit_tests_results.keys()))
                # if all plds are -1, don't add to dictionnary
                if (np.all(min_plds < 0)):
                    continue

                else:
                    min_pld = min(min_plds[min_plds >= 0])
                    groups_fit_test[min_pld] = fit_tests_results[min_pld]

            if (len(groups_fit_test) == {}):
                continue

            else:

                min_pld = min(groups_fit_test.keys())

                params_fits_p.append(groups_fit_test[min_pld][0])
                params_inits_p.append(groups_fit_test[min_pld][1])
                rmses_p.append(groups_fit_test[min_pld][2])
                r2s_p.append(groups_fit_test[min_pld][3])
                peakLocDiffs_p.append(groups_fit_test[min_pld][4])
                gnelds_p.append(groups_fit_test[min_pld][5])
                params_inits_p.append(groups_fit_test[min_pld][6])
                used_angles.append(groups_fit_test[min_pld][7])

        if (peakLocDiffs_p == []):
            peakLocDiffs_p = -1
            params_inits_p = np.array([-1,-1,-1])
            estimates = np.array([-1,-1,-1])
            rmse_val = -1
            r2_val = -1
            peakLocDiff_val = -1
            gneld_val = -1
            selected_prominence_factor = -1
            angles_used = angles_air_

        else:
            peakLocDiffs_p = np.array(peakLocDiffs_p)
            peakLocDiffs_p[(peakLocDiffs_p <= 0) & (~np.isnan(peakLocDiffs_p))] = np.max(peakLocDiffs_p)
            min_idx = np.argmin(peakLocDiffs_p)

            initialization = params_inits_p[min_idx]
            estimates = params_fits_p[min_idx]
            rmse_val = rmses_p[min_idx]
            r2_val = r2s_p[min_idx]
            peakLocDiff_val = peakLocDiffs_p[min_idx]
            gneld_val = gnelds_p[min_idx]
            selected_prominence_factor = prominence_factors[min_idx]
            angles_used = used_angles[min_idx]

        distance_fit_test[peakLocDiff_val] = (estimates, initialization, rmse_val, r2_val, peakLocDiff_val, gneld_val, prominence_, selected_prominence_factor, angles_used[0], angles_used[-1])

    test_plds = np.array(list(distance_fit_test.keys()))
    test_plds = test_plds[test_plds > -1]

    # fitted parameters and information to return
    try:
        info_return = distance_fit_test[min(test_plds)]
    except:
        info_return = distance_fit_test[-1]

    if (verbose == 1):
        print(f"Angles used: [{info_return[8]}, {info_return[9]}] \nRMSE: {info_return[2]} \nR²: {info_return[3]} \nNELD: {info_return[4]} \ngNELD: {info_return[5]}\nProminence: {info_return[6]} \nProminence factor: {info_return[7]} \nInitialization: {info_return[1]} \nFitting: {info_return[0]}")

        # plot fif
        fit_spline = intensity(angles_re_* np.pi / 180, *info_return[0], n, n_ox, n_si, wl, d_ox)
        pld_current = peakloc_diff(dat_,fit_spline, angles_air_, verbose=verbose)
        add_subfigure(angles, data, n, ax6, raw_color, label=raw_label, linewidth=line_width, size=size, markersize=markersize)
        ax6.plot(angles_air_, dat_,label=r"$I_{\rm rs}$", color="tab:blue", linestyle="--", marker="o", linewidth=line_width, markersize=markersize)
        ax6.plot(angles_air_, fit_spline, label=r"$I_{\rm fit}$", color=fit_color, linestyle="--", marker="o", linewidth=line_width, markersize=markersize)
        ax6.vlines([angles_air_[0], angles_air_[-1]], ymin=vline_ymin, ymax=vline_ymax, color=vlines_color, alpha=vline_alpha, linestyles="-", linewidth=vlines_thickness)#, label=r"$\theta_{\rm fit-lim}$")
        ax6.legend(fontsize=size, loc=loc, prop=font_props)
        try:
            fig.suptitle(title_str + f"N={len(angles)}. df=3. RMSE={info_return[2]}, R²={r2_val}\n NELD={info_return[4]} - Prominence={info_return[5]} - Prom_fact={info_return[6]}\nAngles used: {(info_return[7], info_return[8])}\nFound parameters: [a={info_return[0][0]}, b={info_return[0][1]}, h={info_return[0][2]}]", size=size, fontname=font, y=1)
        except:
            print("")
        # make_space_above(np.array([ax1]), topmargin=1)
        fig.tight_layout()

        if (path != ""):
            if (apply_filt):
                fig.savefig(path + f"_smooth_median{filt_window}size.pdf")
                fig.savefig(path + f"_smooth_median{filt_window}size.svg")
            else:
                fig.savefig(path + ".pdf")
                fig.savefig(path + ".svg")

        plt.rcParams['figure.figsize'] = [2, 2]
        plt.show()

    if (len(test_plds) == 0):
        info_return = distance_fit_test[-1]

    if (use_tk):
        return info_return, canvas

    return info_return

def SAIM_reconstruction(path, save_path, angles, angle_range=None, params=None, heights_matrix=None, path_correction="", \
                        roi=[], roi_type='rect', frame=0, \
                        power_correction=False, scipy_fit=False, subtract_bg=False, \
                        num_bg_vals=None, \
                        normalization=False, norm_type ='minmax', \
                        correction=False, interpolate_neg=False, \
                        proper='max_angle', keep_initial_values=False, use_peaks=False, \
                        freeze_params=False, \
                        fif_metric=rmse, fif_thresh=0.1, minimize_rmse=False, \
                        group_size=3, modulation_thresh=0.2, \
                        h_win=2000, h_step=100, second_pass=False, spacing_thresh=0.4, \
                        vis3d=False, d_ox=1000, wl=488, na=1.2, pix_size=121.8, \
                        apply_filt=False, filt_window=3, filt_type='median', savgol_poly=3, \
                        reg_val=0.1, beta1=1/3, beta2=2, max_iterations=100, exit_val=2, fletcher=True, \
                        add_momentum=True, alpha=0.1, verbose=0, \
                        curv_metric='gaussian', colormap='rainbow', option=1):
    '''
    Parameters: - path: location of file containing raw SAIM images.
                - save_path: location to save image of heights.
                - angles: in degrees, list of angles in the imaging medium (water) at which data was taken.
                - angle_range: tuple or array of two elements. They indicate the fist and last
                               angles to consider for the reconstruction. If None, all the angles
                               are taken. Make sure the last and first angles are contained in the
                               variable 'angles'.
                - params: array of three values for initial guess. The values are [a, b, h].
                - heights_matrix: np.array of the same size as the region to recosntruct in the image
                                  containing a height initialization for each pixel.
                - path_correction: location of images to use to correct for intensity
                                   changes on images in path. It is assumed that the
                                   images in path correction were taken from a uniform
                                   fluorescent slide or a dye solution.
                                   It is an empty string by default.
                - roi: np array defining ROI of the form [x_start, y_start, x_length, y_length].
                - roi_type: string, type of the ROI to extract. Two options: 'rect' or 'poly'.
                            'rect': one rectangular ROI.
                            'polys': one or multiple polygonal ROIs.
                - frame: int, in stack it is the slide number to visualize for ROI selection.
                - power_correction: boolean, accounts for differences in laser power for each angle.
                                    The correction values are determined experimentally. And can be
                                    changed in the function power_compensation() from folder imagesManipulation.py
                - scipy_fit: boolean, if selected uses Scipy's least_squares function.
                - subtract_bg: boolean option to subtract background.
                               The user is shown an image to choose an area that will
                               be used to calculate an average value per slice in the
                               stack and those values will be considered as the background
                               to subtract.
                - num_bg_vals: int, number of first minimum values to consider for background at an angle.
                - normalization: boolean, option to normalize data in each pixel.
                - norm_type: string, how to normalize the data. Choose between 'minmax' and 'max'.
                             'minmax': (data - min(data)) / (max(data) - min(data)).
                             'max': data / max(data).
                - correction: boolean to choose if it is necessary to correct the images
                              in path with images in path_correction.
                              Default is False.
                - interpolate_neg: boolean, if chosen interpolates negative pixels by calculating
                                   the mean value of its non-negative neighbors.
                - proper: property to consider in order to determine height initialization.
                          Options are 'max_angle' (max intensity angle),
                          and 'minmax_diff' (absolute difference between angles of
                          max and min intensity). Returns 0 if invalid option given.
                - keep_initial_values: boolean, if True, the fitting is not done and
                                       instead the initialization values are returned.
                - use_peaks: boolean, if True, considers the numbers of peaks in the
                             criterion for initialization.
                - freeze_params: boolean, if True, fit is only to the height parameter and not the
                                 amplitude and bias.
                - fif_metric: default chi_square, function for fitness of fit metric.
                - fif_thresh: threshold to consider height estimation valid,
                              heights for fif_metric higher than threshold are processed again if second_pass is True.
                - minimize_rmse: boolean, default False. Option to minimize the RMSE (optimize FIF) value at each pixel.
                - group_size: int, minimum number of modulations to form a group when optimizing FIF metric.
                - modulation_thresh: float, relative modulation with respect to mean peak to peak amplitude.
                                     Peaks with amplitude bigger than that are considered and the rest ignored.
                - h_win: int, default 2000nm. Window of heights to minimize quality metric.
                - h_step: int, default 100nm. Step size when minimizing quality metric.
                - second_pass: boolean, redo reconstruction for pixels with bad quality metric. Default False.
                               For the moment a bad quality metric is PLD >= 0.1.
                - spacing_thresh: float, used when second_pass is True.
                - vis3D: default False, option to show 3D visulaization.
                - d_ox: nm, oxide layer thickness.
                - wl: nm, excitation wavelength.
                - na: objective's numerical apeture.
                - pix_size: nm, pixel size of the image. Assumed square pixels.
                - apply_filt: Apply a filter to smooth intens_noisy if True.
                              The filtered data is used for the LM algorithm. If False, the
                              raw data is given to the LM algorithm.
                              Default is False.
                - filt_window: filter window.
                               Default is 3 (decided after testing different values).
                - filt_type: type of filter to apply median filter ('median') or
                             Savitzky-Golay filter ('savgol').
                             Default 'median'.
                - savgol_poly: order of the polynomial to fit the samples when using Savgol smoothening.
                               Default is 3.
                - reg_val: regularization initial value, set to 0.1 by default if not indicated by user.
                - beta1: factor to reduce regularization value when a step is accepted,
                         set to 1 / 3 if not indicated by the user. Must be < 1.
                - beta2: factor to increase regularization value if step not accepted,
                         set to 2 when not explicitely given by user.
                - max_iterations: maximum number of iterations if algorithm doesn't converges.
                                  100 is the default value.
                - exit_val: scalar positive, exit condition value.
                - fletcher: boolean, if False, identity matrix used for regularization; if True, Fletcher variant
                            used for regularization (instead of identity uses diag_matrix(J_transpose J));
                            default value True.
                - add_momentum: booelan, True to use momentum correction in the LM algorithm,
                                False otherwise (just first order update). Default is True.
                - alpha: acceptance condition when using momentum, compares second order
                         contribution to first order contribution. Step with momentum accepted
                         when 2*norm(delta_2nd_order)/(norm(delta_1st_order))<= alpha. Set
                         to 0.1 by default.
                - curv_metric: curvature metric to calculate, default gaussian;
                               options: gaussian, mean, principal;
                               principal is a tuple (principal_max, principal_min).
                - colormap:  string, from matplotlib package. Used as colormap to show images.
                - option: choose how to present the data. Two options: 1 or 2,
                          1: shows the 3D representation and 2D projection in the separate figure boxes.
                          2: shows the 3D representation and 2D projection in the same figure box.
    Returns: path where the image result and related information are saved.
    '''
    global pixels
    global polys
    def poly_event(event, x, y, flags, params):
        '''
        Function to chose polygonal ROIs. Modifies the global
        variables pixels and polys.
        '''
        # Stores coordinates of points defining a polygonal ROI
        global pixels
        # Stores multiple polygonal ROI definitions
        global polys
        # checking for right mouse clicks
        if event==cv2.EVENT_LBUTTONDOWN:
            pixels.append((x, y))
            if (len(pixels) > 1):
                x1, y1 = pixels[len(pixels)-2]
                x2, y2 = pixels[len(pixels)-1]
                cv2.line(image, (x1, y1), (x2, y2), (255,255,255), 1)
        # Checking for right mouse clicks
        if event==cv2.EVENT_RBUTTONDOWN:
            if (len(pixels) > 1):
                x1, y1 = pixels[-1]
                x2, y2 = pixels[0]
                cv2.line(image, (x1, y1), (x2, y2), (255,255,255), 1)
                polys.append(np.array(pixels))
                pixels = []

    # check GPU availability
    use_gpu = gf.cuda_available()

    # get Si refractive index
    n_si = silicon_refractive_index(wl)
    # get SiO2 refractive index
    n_ox = silica_refractive_index(wl)
    # get water refractive index
    n_im = h2o_refractive_index(wl)

    # Check that angles is valid
    if (len(angles) < 2):
        print("-----------------------------------")
        print("Error: not a valid array of angles.")
        print("-----------------------------------")

    # Extract images from tif path
    data, _ = open_images(path)

    # Check for correction using reference images
    if (correction == True):
        if (path_correction == ""):
            print("---------------------------------------------------------")
            print("Error: please indicate a valid input for path_correction.")
            print("---------------------------------------------------------")
            return

        # Extract images from tif path
        cor_images, _ = open_images(path_correction)

        # Perform correction
        data = correct_intensity_images(data, cor_images)
        if (data is None):
            print("----------------------------------------")
            print("Error: not a valid set of images images.")
            print("----------------------------------------")
            return

    # select data from angles indicated by the user
    if (np.all(angle_range)):
        # find indexes corresponding to angles
        idx_init = np.argmin(np.abs(angles - angle_range[0]))
        idx_end = np.argmin(np.abs(angles - angle_range[1])) + 1
        # select correct angles
        angles = angles[idx_init: idx_end]
        # select correct data
        data = data[idx_init: idx_end]

    # angles in air
    angle_range_air = np.round(np.arcsin(np.sin(np.array(angle_range) * np.pi / 180) * n_im / 1) * 180 / np.pi, 1)
    angles_air = np.round(np.arcsin(np.sin(np.array(angles) * np.pi / 180) * n_im / 1) * 180 / np.pi, 1)

    # User chooses ROI for data reconstruction
    # rectangular ROI
    if (roi_type == 'rect'):
        if (roi == []):
            showCrosshair = False
            fromCenter = False
            winname = "Select ROI (Press 'c' to select all)"
            cv2.namedWindow(winname, cv2.WINDOW_GUI_NORMAL)
            roi = cv2.selectROI(winname, data[frame] / np.max(data[frame]), showCrosshair, fromCenter)
        # save roi definition
        np.save(save_path[:-4] + "_roi_definition.npy", roi)
        # if no selection was made
        if (np.all(roi) == 0):
            data_roi = data
        else:
            data_roi = data[:, roi[1]:(roi[1]+roi[3]), roi[0]:(roi[0]+roi[2])]

        mask = np.ones_like(data_roi[0])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # User chooses polygonal ROIs
    elif (roi_type == 'polys'):
        if (roi == []):
            showCrosshair = False
            fromCenter = False
            image = data[frame] / np.max(data[frame])
            winname = "Select ROI (Press 'c' twice to continue)"
            cv2.namedWindow(winname, cv2.WINDOW_GUI_NORMAL)
            cv2.imshow(winname, image)
            cv2.setMouseCallback(winname, poly_event)
            while True:
                # display the image and wait for a keypress
                cv2.imshow(winname, image)
                key = cv2.waitKey(1) & 0xFF
                # if the 'r' key is pressed, reset the cropping region
                if key == ord("r"):
                    image = data[frame] / np.max(data[frame])
                # if the 'c' key is pressed, break from the loop
                elif key == ord("c"):
                    break

            cv2.waitKey(0)
            cv2.destroyAllWindows()

        else:
            polys = roi

        if (len(polys) > 0):
            # save ROI definiton
            polys = np.array(polys)
            np.save(save_path[:-4] + "_polygon_roi_definition.npy", polys)
            # create mask with polygon ROIs
            mask = np.zeros_like(data[0])
            mask = cv2.fillPoly(mask, polys, (255,255,255)) / 255
            save_tif(mask, save_path[:-4] + "_poly_mask.tif")
            #mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) / 255
            # apply mask to raw data
            data_roi = np.multiply(data, mask)
            # reduce image to include only polynomial ROIs
            polys = np.concatenate(polys).flatten()
            ys = polys[1::2]
            ys.sort()
            xs = polys[::2]
            xs.sort()
            data_roi = data_roi[:, ys[0]: ys[-1] + 1, xs[0]: xs[-1] + 1]
            # reduce mask image to roi
            mask = mask[ys[0]: ys[-1] + 1, xs[0]: xs[-1] + 1]
            save_tif(mask, save_path[:-4] + "_poly_mask_reduced.tif")
            # reinitialize
            pixels = []
            polys = []

        else:
            data_roi = data

    # arrange data in corresponding increasing angle value
    angles, data_roi = sort_angles_match_images(angles, data_roi)
    # save roi stack
    roi_stack = [Image.fromarray(img) for img in data_roi]
    roi_save = roi_stack[0]
    roi_save.save(save_path[:-4] + "_roi.tif", save_all=True, append_images=roi_stack[1:])

    # Check for laser power correction
    if (power_correction):
        data_roi = power_compensation(data_roi)
    # data shape
    depth, rows, cols = data_roi.shape
    # User chooses ROI for background estimation
    if (subtract_bg):
        if (num_bg_vals):
            # find background values in whole image
            bg = np.array([np.mean(np.sort(slide[slide>0].flatten())[:num_bg_vals]) for slide in data])
            np.save(save_path[:-4] + "_bg-values.npy", bg)
            # subtract background
            data_roi = np.array([img - val for img, val in zip(data_roi, bg)])

        # if no number of values for background, ask for an ROI in image
        else:
            showCrosshair = False
            fromCenter = False
            winname = "Select backgorund area"
            bg_roi = cv2.selectROI(winname, data[0] / np.max(data[0]), showCrosshair, fromCenter)
            bg_data_roi = data[:, bg_roi[1]:(bg_roi[1]+bg_roi[3]), bg_roi[0]:(bg_roi[0]+bg_roi[2])]
            #cv2.waitKey(0)
            cv2.destroyAllWindows()
            # if selection was made
            if (np.all(roi) != 0):
                # save background ROI stack
                bg_stack = [Image.fromarray(img) for img in bg_data_roi]
                bg_save = bg_stack[0]
                bg_save.save(save_path[:-4] + "_bg-roi.tif", save_all=True, append_images=bg_stack[1:])
                # background mean at each slice for subtraction
                bg_thresh = np.mean(np.mean(bg_data_roi, axis=1, dtype=np.float64), axis=1, dtype=np.float64)
                # subtract background
                data_roi = np.array([img - avg for img, avg in zip(data_roi, bg_thresh)])

    # Correct for negative pixels
    if (interpolate_neg):
        _ = [interpolate_pixel(img) for img in data_roi]

    # Reshape data
    datas = data_roi.ravel(order="F").reshape(rows * cols, depth).astype(float)
    print("Area shape: %d x %d pixels"%(rows, cols))
    print(f"Non-null pixels: {len(datas[datas[:,0]> 0])}")

    # Reshape height intialization matrix
    try:
        heights_matrix = heights_matrix.ravel(order='F')
    except:
        heights_matrix = np.array([None] * rows * cols)

    # Calculate heights using least-squares
    # if minimizing RMSE per pixel
    est_zeros = np.array([-1, -1, -1]) # pixel in background, assign this for estimates
    if (minimize_rmse):
        estimates = []
        initializations = []
        rmse_vals = []
        r2_vals = []
        peakDiff_vals = []
        prominences = []
        prom_factors = []
        startAngles = []
        endAngles = []
        global_nelds = []
        # file containing theoretical number of peaks
        # as a function of height in ascending order for height
        i = 0
        for dat in datas:
            if (np.all(dat > 0) & (max(dat) < (2**16))):
                try:
                    params_fits, params_inits, rmse_val, r2_val, peakDiff_val, global_neld, prominence, prominence_factor, angleStart, angleEnd = SAIM_one_pixel(angles, dat, \
                                                                                     n=n_im, n_ox=n_ox, n_si=n_si, wl=wl, d_ox=d_ox, \
                                                                                     scipy_fit=scipy_fit, \
                                                                                     use_gpu=use_gpu, \
                                                                                     normalization=normalization, \
                                                                                     norm_type =norm_type, \
                                                                                     keep_initial_values=keep_initial_values, \
                                                                                     h_init = heights_matrix[i], \
                                                                                     h_win=h_win, \
                                                                                     h_step=h_step, \
                                                                                     group_size=group_size, modulation_thresh=modulation_thresh, \
                                                                                     use_peaks=use_peaks, freeze_params=freeze_params, \
                                                                                     filt_type=filt_type, apply_filt=apply_filt, \
                                                                                     filt_window=filt_window, \
                                                                                     savgol_poly=savgol_poly \
                                                                                    )

                except:
                    print(i, i%rows, i//rows)
                    #plt.plot(angles_air, dat + bg)
                    #plt.show()
                    # return None
                    params_fits = est_zeros
                    params_inits = est_zeros
                    rmse_val, r2_val, peakDiff_val = -1, -1, -1
                    prominence, prominence_factor, angleStart, angleEnd = -1, -1, -1, -1
                    global_neld = -1

                estimates.append(params_fits)
                initializations.append(params_inits)
                rmse_vals.append(rmse_val)
                r2_vals.append(r2_val)
                peakDiff_vals.append(peakDiff_val)
                prominences.append(prominence)
                prom_factors.append(prominence_factor)
                startAngles.append(angleStart)
                endAngles.append(angleEnd)
                global_nelds.append(global_neld)

            else:
                # pixel in background, assign this for estimates
                estimates.append(est_zeros)
                initializations.append(est_zeros)
                rmse_vals.append(-1)
                r2_vals.append(-1)
                peakDiff_vals.append(-1)
                prominences.append(-1)
                prom_factors.append(-1)
                startAngles.append(-1)
                endAngles.append(-1)
                global_nelds.append(-1)

            i += 1
        estimates = np.array(estimates)
        initializations = np.array(initializations)
        print(i)

        startAngles_img = np.array(startAngles).reshape(rows, cols, order="F").astype(float)
        startAngles_img[data_roi[0] == 0] = None
        ing_save = Image.fromarray(startAngles_img)
        ing_save.save(save_path[:-4] + "_" + 'start_angle_final_angle_range.tif')

        endAngles_img = np.array(endAngles).reshape(rows, cols, order="F").astype(float)
        endAngles_img[data_roi[0] == 0] = None
        ing_save = Image.fromarray(endAngles_img)
        ing_save.save(save_path[:-4] + "_" + 'end_angle_final_angle_range.tif')

    # same initialization for all pixels
    else:
        # Initialize data structure to store the parameters after regression
        estimates = np.array([[0,0,0]] * len(datas), dtype='f')
        estimates[datas[:,0] > 0] = [LM_with_momentum(angles, dat, n_im, n_ox, n_si, wl, d_ox, \
                                      scipy_fit=scipy_fit, \
                                      params=params, normalization=normalization, \
                                      norm_type =norm_type, \
                                      proper=proper, keep_initial_values=keep_initial_values, \
                                      use_peaks=use_peaks, freeze_params=freeze_params, \
                                      apply_filt=apply_filt, filt_window=filt_window, \
                                      savgol_poly=savgol_poly \
                                     ) \
                     if (np.all(dat > 0) & (max(dat) < (2**16))) else est_zeros \
                     for dat in datas[datas[:,0] > 0]\
                    ]

    # Amplitude paramater
    estimates_a = estimates[:, 0]
    #print(type(estimates_a))
    # Amplitude to square image
    a_img = estimates_a.reshape(rows, cols, order="F").astype(float)
    a_img[data_roi[0] == 0] = None
    # Save amplitude reconstructed image
    a_save = Image.fromarray(a_img)
    a_save.save(save_path[:-4] + "_amplitude.tif")
    # Bias paramater
    estimates_b = estimates[:, 1]
    # Bias to square image
    b_img = estimates_b.reshape(rows, cols, order="F").astype(float)
    b_img[data_roi[0] == 0] = None
    # Save bias reconstructed image
    b_save = Image.fromarray(b_img)
    b_save.save(save_path[:-4] + "_bias.tif")
    # Heights
    estimates_h = estimates[:, 2]
    # Heights to square image
    rec_img = estimates_h.reshape(rows, cols, order="F").astype(float)
    rec_img[data_roi[0] == 0] = None
    # Save Height reconstructed image
    h_save = Image.fromarray(rec_img)
    h_save.save(save_path[:-4] + "_height.tif")
    # Save after interpolating negative pixels
    rec_interp = np.copy(rec_img)
    interpolate_pixel(rec_interp)
    if (roi_type == 'polys'):
        rec_interp = np.multiply(rec_interp, mask)
    h_save = Image.fromarray(rec_interp)
    h_save.save(save_path[:-4] + "_interpolated.tif")
    # Save initialization
    if (minimize_rmse):
        inits_h = initializations[:, 2]
        inits_img = inits_h.reshape(rows, cols, order="F").astype(float)
        inits_img[data_roi[0] == 0] = None
        h_save = Image.fromarray(inits_img)
        h_save.save(save_path[:-4] + "_h_initializations.tif")

    if (minimize_rmse == False):
        if (normalization == True):
                normalized_datas = [(median_filter(dat, filt_window) - np.min(median_filter(dat, filt_window))) / (np.max(median_filter(dat, filt_window)) - np.min(median_filter(dat, filt_window))) for dat in datas]
                normalized_datas = np.array(normalized_datas)
        else:
            normalized_datas = datas

        # rmse
        rmse_vals = np.zeros(len(datas), dtype='f')
        rmse_vals[datas[:,0]> 0] = [rmse(dat, intensity(angles * np.pi / 180, *estimate, n_im, n_ox, n_si, wl, d_ox), angles) \
                      if (estimate.all() != est_zeros.all()) else 0 \
                      for dat, estimate in zip(normalized_datas[datas[:,0]> 0], estimates[datas[:,0]> 0])
                     ]

        # r2
        r2_vals = np.zeros(len(datas), dtype='f')
        r2_vals[datas[:,0]> 0] = [r2(dat, intensity(angles * np.pi / 180, *estimate, n_im, n_ox, n_si, wl, d_ox), angles) \
                      if (estimate.all() != est_zeros.all()) else 0 \
                      for dat, estimate in zip(normalized_datas[datas[:,0]> 0], estimates[datas[:,0]> 0])
                     ]

        # pld
        peakDiff_vals = np.zeros(len(datas), dtype='f')
        peakDiff_vals[datas[:,0]> 0] = [peakloc_diff(dat, intensity(angles * np.pi / 180, *estimate, n_im, n_ox, n_si, wl, d_ox), angles, verbose=verbose) \
                      if (estimate.all() != est_zeros.all()) else 0 \
                      for dat, estimate in zip(normalized_datas[datas[:,0]> 0], estimates[datas[:,0]> 0])
                     ]

    pld_img = np.array(peakDiff_vals).reshape(rows, cols, order="F").astype(float)
    pld_img[data_roi[0] == 0] = None
    ing_save = Image.fromarray(pld_img)
    ing_save.save(save_path[:-4] + "_" + 'neld.tif')
    mean_pld = np.mean(pld_img[data_roi[0] > 0])
    print(f"Mean NELD: {mean_pld:.5}")

    if (minimize_rmse):
        global_nelds_img = np.array(global_nelds).reshape(rows, cols, order="F").astype(float)
        global_nelds_img[data_roi[0] == 0] = None
        ing_save = Image.fromarray(global_nelds_img)
        ing_save.save(save_path[:-4] + "_" + 'global_neld.tif')
        mean_global_neld= np.mean(global_nelds_img[data_roi[0] > 0])
        print(f"Mean Global NELD: {mean_global_neld:.5}")

    # redo reconstruction for pixels with PLD >= 0.1.
    # For the initilization, the height found for the closest good pixel
    # (PLD < 0.1) is taken.
    if (second_pass):
        #good_plds = np.argwhere((pld_img < fif_thresh) & (pld_img > 0))
        filter_pld = (rec_img == -1) #| (pld_img > fif_thresh)
        bad_plds = np.argwhere(filter_pld)
        # according to simuylation data, set h_win to 500nm
        h_win = 500
        estimates = []
        initializations = []
        rmse_vals = []
        r2_vals = []
        peakDiff_vals = []
        prominences = []
        prom_factors = []
        startAngles = []
        endAngles = []
        global_nelds = []
        for bad in bad_plds:
            # new height initializations for bad pixels
            new_hinit = rec_interp[bad[0], bad[-1]]
            if (mask[bad[0], bad[1]] == 1):
                dat = data_roi[:, bad[0], bad[1]]
                try:
                    params_fits, params_inits, rmse_val, r2_val, peakDiff_val, global_neld, prominence, prominence_factor, angleStart, angleEnd = SAIM_one_pixel(angles, dat, \
                                                                                     n=n_im, n_ox=n_ox, n_si=n_si, wl=wl, d_ox=d_ox, \
                                                                                     scipy_fit=scipy_fit, \
                                                                                     use_gpu=use_gpu, \
                                                                                     normalization=normalization, \
                                                                                     norm_type =norm_type, \
                                                                                     keep_initial_values=keep_initial_values, \
                                                                                     h_init=new_hinit, \
                                                                                     h_win=h_win, \
                                                                                     h_step=h_step, \
                                                                                     second_pass=True, spacing_thresh=spacing_thresh, \
                                                                                     group_size=group_size, modulation_thresh=modulation_thresh, \
                                                                                     use_peaks=use_peaks, freeze_params=freeze_params, \
                                                                                     filt_type=filt_type, apply_filt=apply_filt, filt_window=filt_window, \
                                                                                     savgol_poly=savgol_poly \
                                                                                    )

                except:
                    print(i, i%rows, i//rows)
                    # return None
                    params_fits = est_zeros
                    patams_inits = est_zeros
                    rmse_val, r2_val, peakDiff_val = -1, -1, -1
                    global_neld = -1

            # when bad pixel outside mask area
            else:
                # pixel in background, assign this for estimates
                estimates.append(est_zeros)
                initializations.append(est_zeros)
                rmse_vals.append(-1)
                r2_vals.append(-1)
                peakDiff_vals.append(-1)
                prominences.append(-1)
                prom_factors.append(-1)
                startAngles.append(-1)
                endAngles.append(-1)
                global_nelds.append(-1)
                continue

            estimates.append(params_fits)
            initializations.append(params_inits)
            rmse_vals.append(rmse_val)
            r2_vals.append(r2_val)
            peakDiff_vals.append(peakDiff_val)
            prominences.append(prominence)
            prom_factors.append(prominence_factor)
            startAngles.append(angleStart)
            endAngles.append(angleEnd)
            global_nelds.append(global_neld)

        if (estimates != []):
            estimates = np.array(estimates)
            initializations = np.array(initializations)
            rmse_vals = np.array(rmse_vals)
            r2_vals = np.array(r2_vals)
            peakDiff_vals = np.array(peakDiff_vals)

            # Amplitude paramater
            a_img[filter_pld] = estimates[:, 0]
            a_img[data_roi[0] == 0] = None
            # Save new amplitude reconstructed image
            a_save = Image.fromarray(a_img)
            a_save.save(save_path[:-4] + "_amplitude2.tif")
            # Bias paramater
            b_img[filter_pld] = estimates[:, 1]
            b_img[data_roi[0] == 0] = None
            # Save new bias reconstructed image
            b_save = Image.fromarray(b_img)
            b_save.save(save_path[:-4] + "_bias2.tif")
            # Heights
            rec_img[filter_pld] = estimates[:, 2]
            rec_img[data_roi[0] == 0] = None
            # Save new Height reconstructed image
            h_save = Image.fromarray(rec_img)
            h_save.save(save_path[:-4] + "height2.tif")
            # Save after interpolating negative pixels
            rec_interp = np.copy(rec_img)
            interpolate_pixel(rec_interp)
            if (roi_type == 'polys'):
                rec_interp = np.multiply(rec_interp, mask)
            h_save = Image.fromarray(rec_interp)
            h_save.save(save_path[:-4] + "_interpolated2.tif")
            # Save initialization
            inits_img[filter_pld] = initializations[:, 0]
            inits_img[data_roi[0] == 0] = None
            h_save = Image.fromarray(inits_img)
            h_save.save(save_path[:-4] + "_h_initializations2.tif")

            startAngles_img[filter_pld] = startAngles
            startAngles_img[data_roi[0] == 0] = None
            ing_save = Image.fromarray(startAngles_img)
            ing_save.save(save_path[:-4] + "_" + 'start_angle_final_angle_range2.tif')

            endAngles_img[filter_pld] = endAngles
            endAngles_img[data_roi[0] == 0] = None
            ing_save = Image.fromarray(endAngles_img)
            ing_save.save(save_path[:-4] + "_" + 'end_angle_final_angle_range2.tif')

            pld_img[filter_pld] = peakDiff_vals
            pld_img[data_roi[0] == 0] = None
            ing_save = Image.fromarray(pld_img)
            ing_save.save(save_path[:-4] + "_" + 'neld2' + ".tif")
            mean_pld = np.mean(pld_img[data_roi[0] > 0])
            print(f"Mean NELD: {mean_pld:.5}")

            global_nelds_img[filter_pld] = global_nelds
            global_nelds_img[data_roi[0] == 0] = None
            ing_save = Image.fromarray(global_nelds_img)
            ing_save.save(save_path[:-4] + "_" + 'global_neld2.tif')
            mean_global_neld= np.mean(global_nelds_img[data_roi[0] > 0])
            print(f"Mean Global NELD: {mean_global_neld:.5}")

    # images to show figures
    rec_img[pld_img == -1] = None
    pld_img[pld_img == -1] = None
    if (minimize_rmse):
        global_nelds_img[pld_img == -1] = None
    else:
        global_nelds_img = pld_img

    return save_path, np.array(roi_stack[0]), data_roi[0], rec_img, pld_img, global_nelds_img
