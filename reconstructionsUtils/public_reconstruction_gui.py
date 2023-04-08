import warnings; warnings.simplefilter('ignore')
# import matplotlib
# matplotlib.use('TkAgg')
# import numpy as np
# # import pandas as pd
# import math
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import cm
# from matplotlib.ticker import LinearLocator, FormatStrFormatter
# import matplotlib
# import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import *
from tkinter import ttk
from tkinter.filedialog import askdirectory
import ctypes
# import time
# from matplotlib import ticker
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
# from matplotlib.figure import Figure
# from skimage import io
# import cv2
# interference_equations contains the functions for simulations:
# one_beam_intensity, two_beam_intensity
# from .public_interferenceEquations import *
# from .public_leastSquaresLMmomentum import *
# from .public_evaluationFunctions import *
# from .public_filesManipulation import *
# from .public_visualizationSave import *
# from .public_imagesManipulation import *
# from .public_reconstructions import *
from .public_gui_utils import *

def gui_reconstruction():
    #ctypes.windll.shcore.SetProcessDpiAwareness(1)
    font_section = "Arial"
    section_font_size = 12
    font = "Arial"
    label_font_size = 9
    value_font_size = 9
    bar_label_pad1 = 15
    bar_label_pad2 = 15
    bar_label_rotation = -90

    # Define reference critical angle in degrees for immersion oil
    # 1.47: cell's mounting medium refractive index according to
    # https://www.thermofisher.com/us/en/home/life-science/cell-analysis/cellular-imaging/fluorescence-microscopy-and-immunofluorescence-if/mounting-medium-antifades.html?SID=fr-antifade-main
    # 1.515: immersion oil's refractive index according to
    # http://www.cargille.com/immersionoilmicroscope.shtml
    n_cell = 1.47
    n_oil = 1.515
    critical_angle_deg = math.asin(n_cell / n_oil) * 180 / np.pi

    # Variable to store pixels
    pixels = []

    def clicked_reconstruction():
        '''
        This the function calling the revonstruction functions when
        the Update button in the GUI is pressed or the options for
        Fresnel Coefficient, Phase and PI Shift are changed.
        '''
        select_folder_aux()
        clicked_reconstruction_util(wave, dox, min_theta, max_theta, step_theta, pix_s, num_ap, index_nim, index_nox, index_nsi, \
                                    fold, prosaim_file, roi_t, critical_angle, \
                                    sbg_select, bg_pixs, \
                                    sq_num, rec_time, h_mean, \
                                    mean_neld, inith_seq_select, optional_h_window)

    def clicked_generate_lookup_tables():
        clicked_generate_lookup_tables_util(wave, dox, min_theta, max_theta, step_theta)

    def select_folder_aux():
        '''
        Fills or correct spaces in MAxSIMAcquisition Details panel
        and in Information Panel.
        '''
        folder = fold.get()
        if (len(folder) != 0):

            # read stack file to recosntruct
            saim_path_data = glob.glob(folder + "/*SAIM*mrc")
            if (saim_path_data == []):
                saim_path_data = glob.glob(folder + "/*SAIM*tif")

            if (saim_path_data != []):
                saim_path_data = saim_path_data[0][saim_path_data[0].rfind("/") + 1:]
                fill_reading_space(prosaim_file, saim_path_data)

            # read wavelength
            wl = int(wave.get())
            try:
                wl2 = int(sim_path_data[:3])
            except:
                wl2 = int(saim_path_data[5:8])

            # if user's wavelenght different from file, prefer the one from file
            if (wl != wl2):
                wave.current(wave['values'].index(str(wl2)))

            # clean in information panel
            fill_reading_space(rec_time, '')
            fill_reading_space(sq_num, '')
            fill_reading_space(h_mean, '')

    def select_folder():
        '''
        Captures folder selection into folder path.
        '''
        folder = askdirectory()
        fold.delete(0, END)
        fold.insert(END, folder)

        select_folder_aux()

    def select_ref_idx_medium(eventObject):
        '''
        Assigns refractive indexes according to immersion
        medium or wavelenght.
        '''
        # get medium and tempoerature
        medium = medium_op.get()
        tmprt = tmprt_op.get()
        wl = int(wave.get())

        fill_reading_space(index_nox, "%.4f"%silica_refractive_index(wl))
        fill_reading_space(index_nsi, "%.4f"%silicon_refractive_index(wl))

        if (medium == 'water'):
            #if (tmprt == '23'):
            fill_reading_space(index_nim, "%.4f"%h2o_refractive_index(wl))

        elif (medium == 'glycerine'):
            fill_reading_space(index_nim, '1.456')

        elif (medium == 'oil'):
            fill_reading_space(index_nim, '1.518')

    def enable_bg_pixels(eventObject):
        '''
        Enables space to input number
        of lowest pixels to consider.
        '''
        if (bg_op.get() == 'average n lowest pixels'):
            lbl_low_pixs.config(text='            n:')
            bg_pixs.configure(state='normal')
            bg_pixs.delete(0, END)
            bg_pixs.insert(END, '20')

        else:
            lbl_low_pixs.config(text='              ')
            fill_reading_space(bg_pixs, '')

    def quit():
        '''
        Function to quit and destroy the window.
        '''
        window.destroy()
        window.quit()


    window = Tk()
    # Set window title
    window.title("MAxSIM Reconstruction GUI")
    # Color for backgorund
    bg_color = 'gray30'# 'black'#
    # Color for text
    fg_color = 'white'
    # Color for plots
    plt_color = 'yellow'

    frame_acq_details = Frame(window, bg=bg_color, borderwidth=1, relief='ridge')
    frame_acq_details.master.configure(bg=bg_color)
    frame_acq_details.grid(row=0, column=0, sticky=NW, rowspan=2)

    # Set frame for MAxSIM options
    frame_saim_options = Frame(window, bg=bg_color, borderwidth=1, relief='ridge', width=1000)
    frame_saim_options.master.configure(bg=bg_color)
    frame_saim_options.grid(row=2, column=0, sticky=NW)

    # Set frame for height lookup table generation
    frame_htable_gen = Frame(window, bg=bg_color, borderwidth=1, relief='ridge')
    frame_htable_gen.master.configure(bg=bg_color)
    frame_htable_gen.grid(row=0, column=1, sticky=NW)

    # Set frame for reconstruction
    frame_reconstruction = Frame(window, bg=bg_color, borderwidth=1, relief='ridge')
    frame_reconstruction.master.configure(bg=bg_color)
    frame_reconstruction.grid(row=1, column=1, sticky=NW)

    # Set frame for results
    frame_result = Frame(window, bg=bg_color, borderwidth=1, relief='ridge')
    frame_result.master.configure(bg=bg_color)
    frame_result.grid(row=2, column=1, sticky=NW)

    # # Bottom scrollbar
    # b_scroll = Scrollbar(window, orient='horizontal')
    # b_scroll.grid(row=10)
    # window['yscrollcommand'] = b_scroll.set

    ##############
    # Parameters #
    ##############
    lbl = Label(frame_acq_details, text="MAxSIM Acquisition Details", font=(font_section, section_font_size), fg='white', bg=bg_color)
    lbl.grid(column=0, row=0, padx=46, pady=10, columnspan=3)

    # Objective selector
    lbl = Label(frame_acq_details, text="Objective Magnification:", font=(font, label_font_size), fg=fg_color, bg=bg_color)
    lbl.grid(sticky=E, column=1, row=2, padx=70) # text position
    objective_op = ttk.Combobox(frame_acq_details, width=6, font=(font, value_font_size))
    objective_op['values']= ('63x', '100x')
    objective_op.current(0) #set the selected item
    objective_op.grid(sticky=E, column=2, row=2, padx=4)

    # Numerical aperture
    lbl = Label(frame_acq_details, text="Numerical apperture:", font=(font, label_font_size), fg=fg_color, bg=bg_color)
    lbl.grid(sticky=E, column=1, row=3, padx=70) # text position
    num_ap = Entry(frame_acq_details, width=8) #, state='disabled' disables widget
    num_ap.insert(END, '1.2')
    num_ap.grid(sticky=E, column=2, row=3, padx=4)

    # Pixel size
    lbl = Label(frame_acq_details, text="Pixel size (nm):", font=(font, label_font_size), fg=fg_color, bg=bg_color)
    lbl.grid(sticky=E, column=1, row=4, padx=70) # text position
    pix_s = Entry(frame_acq_details, width=8) #, state='disabled' disables widget
    pix_s.insert(END, '121.8')
    pix_s.grid(sticky=E, column=2, row=4, padx=4)

    # Wavelength window
    lbl = Label(frame_acq_details, text="Wavelength (nm):", font=(font, label_font_size), fg=fg_color, bg=bg_color)
    lbl.grid(sticky=E, column=1, row=5, padx=70) # text position
    wave = ttk.Combobox(frame_acq_details, width=6, font=(font, value_font_size)) #, state='disabled' disables widget
    wave['values']= ('488', '560', '594', '647')
    wave.current(0)
    wave.bind("<<ComboboxSelected>>", select_ref_idx_medium)
    wave.grid(sticky=E, column=2, row=5, padx=4) # widget position

    # Immerson medium
    lbl = Label(frame_acq_details, text="Immersion medium:", font=(font, label_font_size), fg=fg_color, bg=bg_color)
    lbl.grid(sticky=E, column=1, row=6, padx=70) # text position
    medium_op = ttk.Combobox(frame_acq_details, width=6, font=(font, value_font_size))
    medium_op['values']= ('water', 'glycerine', 'oil')
    medium_op.current(0)
    medium_op.bind("<<ComboboxSelected>>", select_ref_idx_medium)
    medium_op.grid(sticky=E, column=2, row=6, padx=4) # widget position

    # Temperature for immersion medium
    lbl = Label(frame_acq_details, text="Temperature immersion medium(°C):", font=(font, label_font_size), fg=fg_color, bg=bg_color)
    lbl.grid(sticky=E, column=1, row=7, padx=70) # text position
    tmprt_op = ttk.Combobox(frame_acq_details, width=6, font=(font, value_font_size)) #, state='disabled' disables widget
    tmprt_op['values']= ('4', '23', '37')
    tmprt_op.current(1)
    tmprt_op.bind("<<ComboboxSelected>>", select_ref_idx_medium)
    tmprt_op.grid(sticky=E, column=2, row=7, padx=4) # widget position

    # Silica thickness window
    lbl = Label(frame_acq_details, text="SiO\u2082 layer thickness (nm):", font=(font, label_font_size), fg=fg_color, bg=bg_color)
    lbl.grid(sticky=E, column=1, row=8, padx=70) # text position
    dox = Entry(frame_acq_details, width=8) #, state='disabled' disables widget
    dox.insert(END, '1000')
    dox.grid(sticky=E, column=2, row=8, padx=4) # widget position

    # Minimum angle
    lbl = Label(frame_acq_details, text="Lower-bound angle (°):", font=(font, label_font_size), fg=fg_color, bg=bg_color)
    lbl.grid(sticky=E, column=1, row=9, padx=70) # text position
    min_theta = Entry(frame_acq_details, width=8) #, state='disabled' disables widget
    min_theta.insert(END, '19')
    min_theta.grid(sticky=E, column=2, row=9, padx=4) # widget position

    # Maximum angle
    lbl = Label(frame_acq_details, text="Upper-bound angle (°):", font=(font, label_font_size), fg=fg_color, bg=bg_color)
    lbl.grid(sticky=E, column=1, row=10, padx=70) # text position
    max_theta = Entry(frame_acq_details, width=8) #, state='disabled' disables widget
    max_theta.insert(END, '53')
    max_theta.grid(sticky=E, column=2, row=10, padx=4) # widget position

    # Step angle
    lbl = Label(frame_acq_details, text="Step size (°):", font=(font, label_font_size), fg=fg_color, bg=bg_color)
    lbl.grid(sticky=E, column=1, row=11, padx=70) # text position
    step_theta = Entry(frame_acq_details, width=8) #, state='disabled' disables widget
    step_theta.insert(END, '0.5')
    step_theta.grid(sticky=E, column=2, row=11, padx=4) # widget position

    # Refractive indexes
    # lbl = Label(frame_param, text="Refractive indeces:\t  ", font=(font, label_font_size), fg=fg_color, bg=bg_color)
    # lbl.grid(sticky=E, column=1, row=2) # text position
    lbl = Label(frame_acq_details, text="Refractive index of medium:", font=(font, label_font_size), fg=fg_color, bg=bg_color)
    lbl.grid(sticky=E, column=1, row=12, padx=70) # text position
    index_nim = Entry(frame_acq_details, readonlybackground=bg_color, fg=fg_color, width=8) #, state='disabled' disables widget
    index_nim.insert(END, '1.3355')
    index_nim.configure(state='readonly')
    index_nim.grid(sticky=E, column=2, row=12, padx=4) # widget position

    lbl = Label(frame_acq_details, text="Refractive index of SiO\u2082:", font=(font, label_font_size), fg=fg_color, bg=bg_color)
    lbl.grid(sticky=E, column=1, row=13, padx=70) # text position
    index_nox = Entry(frame_acq_details, readonlybackground=bg_color, fg=fg_color, width=8) #, state='disabled' disables widget
    index_nox.insert(END, '1.4630')
    index_nox.configure(state='readonly')
    index_nox.grid(sticky=E, column=2, row=13, padx=4) # widget position

    lbl = Label(frame_acq_details, text="Refractive index of Si:", font=(font, label_font_size), fg=fg_color, bg=bg_color)
    lbl.grid(sticky=E, column=1, row=14, padx=70) # text position
    index_nsi = Entry(frame_acq_details, readonlybackground=bg_color, fg=fg_color, width=8) #, state='disabled' disables widget
    index_nsi.insert(END, '4.3676')
    index_nsi.configure(state='readonly')
    index_nsi.grid(sticky=E, column=2, row=14, padx=4) # widget position

    # Critical angle
    lbl = Label(frame_acq_details, text="Critical angle (°):", font=(font, label_font_size), fg=fg_color, bg=bg_color)
    lbl.grid(sticky=E, column=1, row=15, padx=70) # text position
    critical_angle = Entry(frame_acq_details, readonlybackground=bg_color, fg=fg_color, width=8) #, state='disabled' disables widget
    critical_angle.insert(END, '%.2f'%(critical_angle_deg))
    critical_angle.configure(state='readonly')
    critical_angle.grid(sticky=E, column=2, row=15, padx=4) # widget position

    ################
    # MAxSIM OPTIONS #
    ################
    lbl = Label(frame_saim_options, text="MAxSIM Reconstruction Parameters", font=(font_section, section_font_size), fg='white', bg=bg_color)
    lbl.grid(column=0, row=0, pady=10, columnspan=2)

    # Set frame for background options
    frame_bg = Frame(frame_saim_options, bg=bg_color, borderwidth=1, relief='ridge')
    frame_bg.master.configure(bg=bg_color)
    frame_bg.grid(row=1, column=0, columnspan=2)

    # Background option
    sbg_select = BooleanVar(frame_bg)
    sbg_select.set(True) #set check state
    sbg_state = Checkbutton(frame_bg, text='Background  \nsubtraction   ', var=sbg_select, offvalue=False, onvalue=True, \
                            font=(font, label_font_size), justify=LEFT, selectcolor=bg_color, fg=fg_color, bg=bg_color, width=15)
    sbg_state.grid(sticky=E, column=0, row=0, pady=8, rowspan=2)

    # Background to subtract
    bg_op = ttk.Combobox(frame_bg, width=23, font=(font, 10))
    bg_op['values']= ('average n lowest pixels', 'ROI mean')
    bg_op.current(0) #set the selected item
    bg_op.bind("<<ComboboxSelected>>", enable_bg_pixels)
    bg_op.grid(sticky=E, column=1, row=0, padx=4, columnspan=2)

    # Number of lowest pixels for background average
    lbl_low_pixs = Label(frame_bg, text="            n:", font=(font, label_font_size), fg=fg_color, bg=bg_color)
    lbl_low_pixs.grid(sticky=E, column=1, row=1, padx=72) # text position
    bg_pixs = Entry(frame_bg, readonlybackground=bg_color, width=8) #, state='disabled' disables widget
    bg_pixs.insert(END, '20')
    bg_pixs.grid(sticky=W, column=2, row=1, padx=4) # widget position

    # Set frame for using an existing reconstruction for initialization
    frame_prev_rec = Frame(frame_saim_options, bg=bg_color, borderwidth=1, relief='ridge')
    frame_prev_rec.master.configure(bg=bg_color)
    frame_prev_rec.grid(row=2, column=0, columnspan=2)

    # Use previous reconstruction
    inith_seq_select = BooleanVar(frame_prev_rec)
    inith_seq_select.set(True) #set check state
    inith_seq_state = Checkbutton(frame_prev_rec, text='Use previous\nreconstruction', var=inith_seq_select, offvalue=False, onvalue=True, \
                               font=(font, label_font_size), justify=LEFT, selectcolor=bg_color, fg=fg_color, bg=bg_color, width=15)
    inith_seq_state.grid(sticky=E, column=0, row=0)

    # Height window for NELD search
    lbl = Label(frame_prev_rec, text="h° range (nm):", font=(font, label_font_size), fg=fg_color, bg=bg_color)
    lbl.grid(sticky=E, column=1, row=0, padx=72) # text position
    optional_h_window = Entry(frame_prev_rec, width=8) #, state='disabled' disables widget
    optional_h_window.insert(END, '1000')
    optional_h_window.grid(sticky=W, column=2, row=0, padx=4) # widget position

    #################################
    # show height lookup table area #
    #################################
    lbl = Label(frame_htable_gen, text="Lookup Table Generation", font=(font_section, section_font_size), fg='white', bg=bg_color)
    lbl.grid(column=0, row=0, padx=80, pady=10, columnspan=3)

    # Generate lookup tables button
    btn = tk.Button(frame_htable_gen, text="Generate\ntables", command=clicked_generate_lookup_tables, bg="DarkOrange3", fg=fg_color, font=(font, label_font_size))
    btn.grid(column=1, row=1, padx=4, pady=10) # button position

    # space
    lbl = Label(frame_htable_gen, text="", font=(font, label_font_size), fg=fg_color, bg=bg_color)
    lbl.grid(sticky=E, column=0, row=3, columnspan=3) # text position

    ##################
    # Reconstruction #
    ##################
    lbl = Label(frame_reconstruction, text="Height Reconstruction", font=(font_section, section_font_size), fg='white', bg=bg_color)
    lbl.grid(column=0, row=0, padx=91, pady=10, columnspan=4)

    # folder selction
    lbl = Label(frame_reconstruction, text="Select folder:", font=(font, label_font_size), fg=fg_color, bg=bg_color)
    lbl.grid(sticky=E, column=0, row=1) # text position
    fold = Entry(frame_reconstruction, width=22) #, state='disabled' disables widget
    fold.insert(END, '') #/home/pfgr/Work/data/7nostripSPSAIM560-GCbottom-488phall-560BCR_20210830_151402
    fold.grid(sticky=W, column=1, row=1, padx=4, columnspan=2) # widget position
    # Choose folder button, call tk.Button instead of Button
    # because that name is taken from .public_visualizationSave import *
    btn = tk.Button(frame_reconstruction, text="Select\nFolder", command=select_folder, bg=fg_color, fg=bg_color, font=(font, 8))
    btn.grid(sticky=W, column=3, row=1, padx=4) # button position

    # ROI type
    lbl = Label(frame_reconstruction, text="ROI type:", font=(font, label_font_size), fg='white', bg=bg_color)
    lbl.grid(sticky=E, column=0, row=2)
    roi_t = ttk.Combobox(frame_reconstruction, width=11, font=(font, 10))
    roi_t['values']= ('rectangular', 'polygonal')
    roi_t.current(0) #set the selected item
    roi_t.grid(sticky=W, column=1, row=2, padx=4)

    # button to launch reconstruction
    lbl = Label(frame_reconstruction, text="Reconstruction:", font=(font, label_font_size), fg=fg_color, bg=bg_color)
    lbl.grid(sticky=E, column=0, row=3) # text position
    # Run reconstruction button, call tk.Button instead of Button
    # because that name is taken by traits.api from .public_visualizationSave import *
    btn = tk.Button(frame_reconstruction, text="Run", command=clicked_reconstruction, bg="green4", fg=fg_color, font=(font, label_font_size))
    btn.grid(sticky=W, column=1, row=3, padx=4, pady=10) # button position

    # Close button, call tk.Button instead of Button
    # because that name is taken traits.api from .public_visualizationSave import *
    btn = tk.Button(frame_reconstruction, text="Quit", command=quit, bg="red4",fg=fg_color, font=(font, label_font_size))
    btn.grid(sticky=W, column=2, row=3, padx=4) # button position

    # space
    lbl = Label(frame_reconstruction, text="", font=(font, label_font_size), fg=fg_color, bg=bg_color)
    lbl.grid(sticky=E, column=0, row=4, pady=1) # text position

    #########################
    # show information area #
    #########################
    lbl = Label(frame_result, text="Information Panel", font=(font_section, section_font_size), fg='white', bg=bg_color)
    lbl.grid(column=0, row=0, pady=10, columnspan=2)

    # Processed file
    lbl = Label(frame_result, text="File name:", font=(font, label_font_size), fg=fg_color, bg=bg_color)
    lbl.grid(sticky=E, column=0, row=1, padx=67) # text position
    prosaim_file = Entry(frame_result, readonlybackground=bg_color, fg=fg_color, width=8) #, state='disabled' disables widget
    prosaim_file.insert(END, '')
    prosaim_file.configure(state='readonly')
    prosaim_file.grid(sticky=E, column=1, row=1, padx=4) # widget position

    # Sequence number
    lbl = Label(frame_result, text="Number of reconsructed images:", font=(font, label_font_size), fg=fg_color, bg=bg_color)
    lbl.grid(sticky=E, column=0, row=2, padx=67) # text position
    sq_num = Entry(frame_result, readonlybackground=bg_color, fg=fg_color, width=8) #, state='disabled' disables widget
    sq_num.insert(END, '')
    sq_num.configure(state='readonly')
    sq_num.grid(sticky=E, column=1, row=2, padx=4) # widget position

    # Reconstruction time
    lbl = Label(frame_result, text="Elapsed time (s):", font=(font, label_font_size), fg=fg_color, bg=bg_color)
    lbl.grid(sticky=E, column=0, row=3, padx=67) # text position
    rec_time = Entry(frame_result, readonlybackground=bg_color, fg=fg_color, width=8) #, state='disabled' disables widget
    rec_time.insert(END, '')
    rec_time.configure(state='readonly')
    rec_time.grid(sticky=E, column=1, row=3, padx=4) # widget position

    # Average Height
    lbl = Label(frame_result, text="Average height (nm):", font=(font, label_font_size), fg=fg_color, bg=bg_color)
    lbl.grid(sticky=E, column=0, row=4, padx=67) # text position
    h_mean = Entry(frame_result, readonlybackground=bg_color, fg=fg_color, width=8) #, state='disabled' disables widget
    h_mean.insert(END, '')
    h_mean.configure(state='readonly')
    h_mean.grid(sticky=E, column=1, row=4, padx=4) # widget position

    # Mean NELD
    lbl = Label(frame_result, text="Mean NELD:", font=(font, label_font_size), fg=fg_color, bg=bg_color)
    lbl.grid(sticky=E, column=0, row=5, padx=67) # text position
    mean_neld = Entry(frame_result, readonlybackground=bg_color, fg=fg_color, width=8) #, state='disabled' disables widget
    mean_neld.insert(END, '')
    mean_neld.configure(state='readonly')
    mean_neld.grid(sticky=E, column=1, row=5, padx=4) # widget position


    window.mainloop()
