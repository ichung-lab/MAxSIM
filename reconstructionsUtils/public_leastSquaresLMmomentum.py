################
## Librairies ##
################

import numpy as np
from scipy.signal import find_peaks, argrelextrema
from scipy.signal import savgol_filter, medfilt
from scipy.optimize import least_squares
from scipy.ndimage import median_filter
from time import time
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib
import matplotlib.pyplot as plt

from .public_interferenceEquations import *
from .public_evaluationFunctions import rmsea, chi_square
from .public_imagesManipulation import power_compensation

########################
## Intensity equation ##
########################

def intensity(angle, a, b, h, n, n_ox, n_si, wl, d_ox=1000, x=0, phi=0, theo_intens=one_beam_intensity):
    '''
    Calculates theoretical intensity according to theoretical model.
    Parameters:
        angle: in radians, incidence angle to the sample.
        a: coefficient applied to thoretical intensity.
        b: represent background noise added to theoretical intensity.
        h: height from the SiO2 surface.
        n: refractive index of sample or immersion media.
        n_ox: refractive index SiO2 layer.
        n_si: refractive index Si layer.
        wl: beam's wavelenght in nm.
        d_ox: depth of SiO2 layer in nm. Default 1000nm.
        x: lateral position, applies only for two-beam interference. Default is 0.
        phi: phase of straignt beam with respect to the other two. Applies for 3-beam intensity. Default is 0.
        theo_intens: theoretical intensity function.
                       Default is one_beam_intensity(n, n_ox, n_si, angle, wl, d_ox, h).
    Returns: intensity for given h, a and b.
    '''

    if (theo_intens == one_beam_intensity):
        I = theo_intens(n, n_ox, n_si, angle, wl, d_ox, h)

    elif (theo_intens == two_beam_intensity):
        I = theo_intens(n, n_ox, n_si, angle, wl, d_ox, h, x)

    elif (theo_intens == three_beam_intensity_phi):
        I = theo_intens(n, n_ox, n_si, angle, wl, d_ox, h, x, phi)

    elif (theo_intens == three_beam_no_reflection):
        I = theo_intens(n, n_ox, n_si, angle, wl, h, x, phi)

    return a * I + b

###########################
## Equation for residual ##
###########################

def residual(parameters, angle, inten, n, n_ox, n_si, wl, d_ox, x=0, \
             intens_func=intensity, theo_intens=one_beam_intensity):
    '''
    Calculates residual (difference) between data and theoretical model.
    Parameters:
        parameters: vector [a, b, h].
        angle: in radians, incidence angle to the sample.
        inten: experimental data.
        n: refractive index of sample or immersion media.
        n_ox: refractive index SiO2 layer.
        n_si: refractive index Si layer.
        wl: beam's wavelenght in nm.
        d_ox: depth of SiO2 layer in nm.
        x: lateral position, applies only for two-beam interference. Default is 0.
        intens_func: function of the theoretical model.
                     Default is
                     intensity(angle, a, b, h, n, n_ox, n_si, wl, d_ox, x=0, theo_intens=one_beam_intensity).
        theo_intens: theoretical intensity function.
                     Default is one_beam_intensity(n, n_ox, n_si, angle, wl, d_ox, h).
    Returns: difference between experimental data and model.
    '''
    # if first two parameters are frozen
    if (len(parameters) == 1):
        a = 1 - np.min(inten)
        b = np.min(inten)

        return (inten - intens_func(angle, a, b, parameters, n, n_ox, n_si, wl, d_ox, x, theo_intens)).flatten()

    elif (len(parameters) == 3):
        return (inten - intens_func(angle, *parameters, n, n_ox, n_si, wl, d_ox, x, theo_intens)).flatten()

def frozen_residual(h, angle, inten, n, n_ox, n_si, wl, d_ox,  a=1, b=0, x=0, \
                    intens_func=intensity, theo_intens=one_beam_intensity):
    '''
    Calculates residual between data and theoretical model by keeping a nd b constant.
    Parameters:
        h: height.
        angle: in radians, incidence angle to the sample.
        inten: experimental data.
        n: refractive index of sample or immersion media.
        n_ox: refractive index SiO2 layer.
        n_si: refractive index Si layer.
        wl: beam's wavelenght in nm.
        d_ox: depth of SiO2 layer in nm.
        a: amplitude for intensity.
        b: bias for intensity.
        x: lateral position, applies only for two-beam interference. Default is 0.
        intens_func: function of the theoretical model.
                     Default is
                     intensity(angle, a, b, h, n, n_ox, n_si, wl, d_ox, x=0, theo_intens=one_beam_intensity).
        theo_intens: theoretical intensity function.
                     Default is one_beam_intensity(n, n_ox, n_si, angle, wl, d_ox, h).
    Returns: difference between experimental data and model.
    '''

    return None

############################
## Derivative of residual ##
############################

def derivative_residual(angle, parameters, n, n_ox, n_si, wl, d_ox, x=0, theo_intens=one_beam_intensity):
    '''
    Calculates derivatives (Jacobian) of the residual with respect to the parameters
    for the model f = a * I(h) + b.
    Parameters:
        angle: in radians, incidence angle to the sample.
        parameters: vector [a, b, h].
        n: refractive index of sample or immersion media.
        n_ox: refractive index SiO2 layer.
        n_si: refractive index Si layer.
        wl: beam's wavelenght in nm.
        d_ox: depth of SiO2 layer in nm.
        x: lateral position, applies only for two-beam interference. Default is 0.
        theo_intens: theoretical intensity function.
                       Default is one_beam_intensity(n, n_ox, n_si, angle, wl, d_ox, h).
    Returns: Jacobian matrix.
    '''
    a = parameters[0]
    b = parameters[1]
    h = parameters[2]

    intens = theo_intens(n, n_ox, n_si, angle, wl, d_ox, h)
    phase_diff = phase(n, angle, wl, h)
    r = rte(n, n_ox, n_si, angle, wl, d_ox)
    temp = r * np.exp(1J * phase_diff)

    der = np.zeros((len(angle), len(parameters)))
    # derivatives with respect to parameters
    der[:, 0] = -intens
    der[:, 1] = -1
    der[:, 2] = a * 8 * np.pi * n * np.cos(angle) * np.imag(temp) / wl

    # der = []
    # derivatives with respect to parameters
    # der.append(-intens)
    # der.append(-1)
    # der.append(a * 8 * np.pi * n * np.cos(angle) * np.imag(temp) / wl)

    return np.array(der)

###############################################
## Directional second derivative of residual ##
###############################################

def directional_second_derivative_residual(angle, params, delta_params, n, n_ox, n_si, wl, d_ox):
    '''
    Calculates the Hessian matrix and multiplies its elements with the
    proper delta_parameters for a data point for the interference produced
    by one beam interference.
    Parameters:
        angle: in radians, incidence angle to the sample.
        parameters: vector [a, b, h].
        delta_params: vector containing the first order correction (using the Jacobian).
        n_ox: refractive index SiO2 layer.
        n_si: refractive index Si layer.
        wl: beam's wavelenght in nm.
        d_ox: depth of SiO2 layer in nm.
    Returns: tensor corresponding to directioanl second derivative of the residual.
    '''
    a = params[0]
    b = params[1]
    h = params[2]
    delta_a = delta_params[0]
    delta_b = delta_params[1]
    delta_h = delta_params[2]

    phase_diff = phase(n, angle, wl, h)
    r = rte(n, n_ox, n_si, angle, wl, d_ox)
    temp = r * np.exp(1J * phase_diff)

    hessian = np.zeros((len(angle), 3, 3))
    aux = np.power(8 * np.pi * n * np.cos(angle) / wl, 2)
    hessian[:, 0, 2] = delta_a * delta_h * aux * np.imag(temp)
    hessian[:, 2, 0] = hessian[:, 0, 2]
    hessian[:, 2, 2] = np.power(delta_h, 2) * aux * np.real(temp)

    return hessian

###########################
## Height initialization ##
###########################

def find_h_init(angles, inten, d_ox, proper='max_angle', use_peaks=False, verbose=0):
    '''
    Determines a height using the parameters p and b,
    for freq = p * height + b, found for the fitting
    of the angular frequency as a function of height.
    Parameters:
        angles: in radians, incidence angles to the sample.
        inten: experimental data.
        d_ox: depth of SiO2 layer in nm.
        proper: property to consider in order to determine
                height initialization.
                Options are 'max_angle' (angle for angle of max intensity),
                and 'minmax_diff' (absolute difference between angles of
                max and min intensity). Returns 0 if invalid option given.
        use_peaks: boolean, if True, considers the numbers of peaks in the
                   criterion for initialization.
        verbose: prints information if 1. Set to 0 by default.
    Returns: initialization for the height.
    '''

    # window and polynomial order for svgol filtering
    window, order = 7, 3
    data = savgol_filter(inten, window, order) #inten #median_filter(inten, neighbors)

    # file containing theoretical peaks and min max differences
    # as a function of height in ascending order for height
    df = pd.read_csv("./reconstructionsUtils/auxiliaryFiles/height_angle_properties_table_19to53degs.csv")
    # number of peaks ot valleys in df
    num = (len(df.columns) - 1) // 2

    # find closest angle to max_angle
    if (proper == 'max_angle'):
        # find angle for peak location in smoothed data
        max_angle = angles[np.argmax(data)]
        # find closest height
        loc = np.argmin(np.abs(df.water_angle_rad_max_intensity - max_angle))

        return df.height_nm[loc]

    elif (proper == 'minmax_diff'):
        # valleys in data
        valleys_ids = find_peaks(np.max(data) - data)[0]
        valleys = np.zeros(num)
        if (len(valleys_ids) >  num):
            valleys = angles[valleys_ids[: num ]]
        else:
            valleys[:len(valleys_ids)] = angles[valleys_ids]
        # peaks in data
        peaks_ids = find_peaks(data)[0]
        peaks = np.zeros(num)
        if (len(peaks_ids) >  num):
            peaks = angles[peaks_ids[: num]]
        else:
            peaks[:len(peaks_ids)] = angles[peaks_ids]

        # calculate criterion
        criterion = np.sum(np.abs(df.iloc[:, 1: num + 1] - peaks), axis=1 ) + np.sum(np.abs(df.iloc[:, num + 1 :] - valleys), axis=1)
        loc = criterion.idxmin()

        return df.height_nm[loc]

    # if invalid option, return 0
    return 0

###############################
## Parameters initialization ##
###############################

def initialize_params(angle, intens, d_ox, h_initialize=find_h_init, proper='minmax_diff', use_peaks=False, verbose=0):
    '''
    Finds the initial parameters: a, b and h.
    Parameters:
        angle: in radians, incidence angle to the sample.
        intens: normalized intensity between 0 and 1.
        d_ox: depth of SiO2 layer in nm.
        h_initialize: function to initialize the height h,
                      by default set to find_h_init(angle, inten, verbose=0).
        proper: property to consider in order to determine
                height initialization.
                Options are 'max_angle' (angle for angle of max intensity),
                and 'minmax_diff' (absolute difference between angles of
                max and min intensity). Returns 0 if invalid option given.
        use_peaks: boolean, if True, considers the numbers of peaks in the
                   criterion for initialization.
        verbose: to pass to h_initialize when necessary. Default is 0.
    Returns: vector of initial parameters.
    '''
    # Parameters assuming the data is normalized between 0 and 1.
    #a = 0.5
    a = (np.max(intens) - np.min(intens)) / 3
    #b = 0
    b = np.min(intens)
    # Initial height
    h_init = h_initialize(angle, intens, d_ox, proper, use_peaks, verbose)

    return [a, b, h_init]


########################################
## LM least-square (GN) with momentum ##
########################################

def LM_with_momentum(theta, intens_noisy, n, n_ox, n_si, wl, d_ox, params=None, x=0, \
                     power_correction=False, \
                     scipy_fit=False, \
                     normalization=False, \
                     norm_type ='minmax', \
                     apply_filt=False, \
                     filt_window=5, \
                     filt_type='median',
                     savgol_poly=3, \
                     freeze_params=False, \
                     intens_func=intensity, \
                     theo_intens=one_beam_intensity, residual_func=residual, \
                     jac_res=derivative_residual, \
                     hess_res=directional_second_derivative_residual, \
                     h_initialize=find_h_init, initial_params=initialize_params, \
                     proper='max_angle', keep_initial_values=False, \
                     use_peaks=False, reg_val=0.1, \
                     beta1=1/3, beta2=2, max_iterations=100, exit_val=2, fletcher=True, \
                     add_momentum=True, alpha=0.1, verbose=0):
    '''
    Levenberg-Marquardt (LM) least-squares algorithm
    with momentum correction.
    Parameters:
        theta: values in degrees for which intens_noisy was acquired.
        intens_noisy: data to fit.
        n: refractive index of sample or immersion media.
        n_ox: refractive index SiO2 layer.
        n_si: refractive index Si layer.
        wl: beam's wavelenght in nm.
        d_ox: depth of SiO2 layer in nm.
        params: array of three values for initial guess.
                  The values are [a, b, h].
        x: lateral position, applies only for two-beam interference. Default is 0.
        power_correction: boolean, accounts for differences in laser power for each angle.
                          The correction values are determined experimentally.
        scipy_fit: boolean, if selected uses Scipy's least_squares function.
        normalization: boolean, option to normalize data.
        norm_type: string, how to normalize the data. Choose between 'minmax' and 'max'.
                   'minmax': (data - min(data)) / (max(data) - min(data)).
                   'max': data / max(data).
        apply_filt: Apply a filter to smooth intens_noisy if True.
                    The filtered data is used for the LM algorithm. If False, the
                    raw data is given to the LM algorithm.
                    Default is False.
		filt_window: filter window.
		             Default is 5.
		filt_type: type of filter to apply median filter ('median') or
		           Savitzky-Golay filter ('savgol').
		           Default 'median'.
		savgol_poly: order of the polynomial to fit the samples when using 'savgol' filter.
					 Default is 3.
        freeze_params: boolean, if True fit is applied only to the third parameter (height).
        intens_func: function of the theoretical model.
                     Default is
                     intensity(angle, theo_intens=one_beam_intensity,a, b, h, n, n_ox, n_si, wl, d_ox, x=0).
        theo_intens: theoretical intensity function.
                     Default is one_beam_intensity(n, n_ox, n_si, angle, wl, d_ox, h).
        residual_func: corresponds to function y = data - theoretical_value.
                       Default is the function
                       residual(parameters, angle, inten, n, n_ox, n_si, wl, d_ox, x=0, \
                                intens_func=intensity, theo_intens=one_beam_intensity).
        jac_res: Jacobian of residual function; default is the function
                 derivative_residual(angle, parameters, n, n_ox, n_si, wl, d_ox, x=0, \
                                     theo_intens=one_beam_intensity).
        hess_res: Hessian of residual funtion; default is the function
                  directional_second_derivative_residual(angle, params, delta_params, \
                                                         n, n_ox, n_si, wl, d_ox).
        h_initialize: function to initialize the height h,
                      by default set to find_h_init(angle, inten, verbose=0).
        initial_params: function producing initial guess for parameters;
                        by default set to
                        initialize_params(angle, intens, h_initialize=find_h_init, verbose=0).
        proper: property to consider in order to determine height initialization.
                Options are 'max_angle' (angle for angle of max intensity),
                and 'minmax_diff' (absolute difference between angles of
                max and min intensity). Returns 0 if invalid option given.
        keep_initial_values: boolean, if True, the fitting is not done and
                             instead the initialization values are returned.
        use_peaks: boolean, if True, considers the numbers of peaks in the
                   criterion for initialization with h_initialize.
        reg_val: regularization initial value, set to 0.1 by default if not
                 indicated by user.
        beta1: factor to reduce regularization value when a step is accepted,
               set to 1 / 3 if not indicated by the user. Must be < 1.
        beta2: factor to increase regularization value if step not accepted,
               set to 2 when not explicitely given by user.
        max_iterations: maximum number of iterations if algorithm doesn't converges.
                        100 is the default value.
        exit_val: scalar positive, exit condition value.
        fletcher: boolean, if False, identity matrix used for regularization; if True, Fletcher variant
                  used for regularization (instead of identity uses diag_matrix(J_transpose J));
                  default value True.
        add_momentum: booelan, True to use momentum correction in the LM algorithm,
                      False otherwise (just first order update). Default is True.
        alpha: acceptance condition when using momentum, compares second order
               contribution to first order contribution. Step with momentum accepted
               when 2*norm(delta_2nd_order)/(norm(delta_1st_order))<= alpha. Set
               to 0.1 by default.
    Returns: [a, b, h] where a is the amplitude, b the bias and h the height the least-square algorithm.
    '''

    # Ignore data if has any negative values
    # or if detector was saturated (assumed to be 16-bit)
    if (np.any(intens_noisy < 0) | np.any(max(intens_noisy) == (2**16 - 1))):
        return np.array([0,0,0])

    # Turn angles to radians
    theta_deg = theta.copy()
    theta = theta * np.pi / 180

    # Check for intensity compensation
    if (power_correction):
        if len(intens_noisy) != 30:
            print("------------------------------------------------------------------")
            print("Error: number of data points doesn't match power calibration data.")
            print("------------------------------------------------------------------")

            return

        else:
            intens_noisy = power_compensation(intens_noisy)

    # Check if data filtering selected for LM algorithm
    if (apply_filt):
        # Smooth data
        if (filt_type == 'median'):
            smooth_intens_noisy = medfilt(intens_noisy, filt_window)

        elif (filt_type == 'savgol'):
            smooth_intens_noisy = savgol_filter(intens_noisy, filt_window, savgol_poly)

        else:
            print("------------------------------------------")
            print("Error: invalid filter type to smooth data.")
            print("------------------------------------------")

        intens_noisy = smooth_intens_noisy

    # Normalize data bewteen 0 and 1
    if (normalization):
        if (norm_type == 'minmax'):
            intens_noisy = (intens_noisy - np.min(intens_noisy)) / (np.max(intens_noisy) - np.min(intens_noisy))
        else:
            intens_noisy = intens_noisy / np.max(intens_noisy)

    # If no initialization indicated by the user, then guess initial parameters
    # Initial guess for parameter a, b and h in f=aI(h)+b
    if (np.all(params) == None):
        params = np.array(initial_params(theta, intens_noisy, d_ox, h_initialize, proper, use_peaks, verbose))

    # return initialization values instead of fitting if chosen to do so
    if (keep_initial_values):
        return  params

    # Check for Scipy's least_square option
    if (scipy_fit):
        # if chosen to fit only the third parameter
        if (freeze_params):
            a = 1 - np.min(intens_noisy)
            b = np.min(intens_noisy)
            h = least_squares(residual_func, params[2], method='lm', args=(theta, intens_noisy, n, n_ox, n_si, wl, d_ox)).x[0]

            return np.array([a, b, h])

        # Apply least_squares
        return least_squares(residual_func, params, method='lm', args=(theta, intens_noisy, n, n_ox, n_si, wl, d_ox)).x

    else:
        # Set initial conditions and values for LM algorithm
        prev_params = params
        #params = np.array(params).transpose()
        # Initialization residuals vector
        res = np.zeros(len(theta))
        # Intialization for Jacobian matrix,
        # a line per data point, a column per parameter
        jacobian = np.zeros((len(theta), len(params)))
        # Initialize identity matrix
        identity = np.identity(len(params))
        # Fill residuals vector
        res = residual_func(params, theta, intens_noisy, n, n_ox, n_si, wl, d_ox, x, \
                            intens_func, theo_intens)
        # Calculate function to minimize
        res_sum = np.sum(np.power(res, 2))
        # Initially, gradient of h is unknown
        gradient_h = np.float("inf")
        # Auxiliary counters
        count1 = 0
        count2 = 0
        rmsea_value = float('inf')
        while (rmsea_value > exit_val):#(gradient_h > 0.000001):# | (params[2] < 0)):
            #inten = one_beam_intensity(nim[i], nox[i], nsi[i], theta, wls[i], d_ox, params[-1])
            # Fill Jacobian matrix
            jacobian = jac_res(theta, params, n, n_ox, n_si, wl, d_ox, x, theo_intens)
            # Transpose of Jacobian
            jacobianT = np.transpose(jacobian)

            jacTjac = np.matmul(jacobianT, jacobian)
            if (fletcher):
                # Fletcher variant
                regularization = reg_val * np.diag(np.diag(jacTjac))

            else:
                # Standard LM update
                regularization = reg_val * identity

            # Damped jacTjac
            damped_jacTjac = np.linalg.inv(jacTjac + regularization)
            # First order update step
            delta_params_1 = - np.matmul(damped_jacTjac, np.matmul(jacobianT, res))

            if (add_momentum):
                # Second order update step
                grad_2_res = hess_res(theta, params, delta_params_1, n, n_ox, n_si, wl, d_ox)
                delta_params_2 = - 0.5 * np.tensordot(damped_jacTjac, np.tensordot(jacobianT, grad_2_res, axes=1), axes=2)
            else:
                delta_params_2 = 0

            # Update params
            new_params = params + delta_params_1 + delta_params_2

            # Calculate new residual and minimization function value
            new_res = residual_func(new_params, theta, intens_noisy, n, n_ox, n_si, wl, d_ox, x, \
                                    intens_func, theo_intens)
            new_res_sum = np.sum(np.power(new_res, 2))
            #print("new_res_sum: {}, res_sum: {}".format(new_res_sum, res_sum))
            # Acceptance condition with mometum: new minimization < old minimization
            # and |delta_params_2| / |delta_params_1| <= alpha
            if (add_momentum) & \
               (2 * np.linalg.norm(delta_params_2) / np.linalg.norm(delta_params_1) <= alpha) & \
               (new_res_sum < res_sum):
                # Update regularization value
                reg_val *= beta1
                # Update res_sum
                res_sum = new_res_sum
                # Update params
                params = new_params
                # Update res
                res = new_res
                count1 += 1
                # Update exit condition
                gradient_h = np.abs(2 * np.matmul(jacobianT, res)[2])

            # Acceptance condition without momentum: new minimization < old minimization
            elif (add_momentum == False) & (new_res_sum < res_sum):
                # Update regularization value
                reg_val *= beta1
                # Update res_sum
                res_sum = new_res_sum
                # Update params
                params = new_params
                # Update res
                res = new_res
                count1 += 1
                # Update exit condition
                gradient_h = np.abs(2 * np.matmul(jacobianT, res)[2])

            # Step not accepted
            else:
                # params no updated
                # Update regularization value
                reg_val *= beta2
                count2 += 1

            # Stop if too many operations
            if (count1 + count2) == max_iterations:
                break

            # Evaluate badness of fit with current patameters
            rmsea_value = rmsea(intens_noisy, intensity(theta_deg, *params, n, n_ox, n_si, wl, d_ox))

        return params
