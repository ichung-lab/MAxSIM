import numpy as np
from scipy.signal import find_peaks
from scipy.stats import mode
from sklearn.metrics import r2_score

# Fitness of fit functions
def peakloc_diff(observed, expected, angles, verbose=0):
    '''
    Calculates the following:
    sum(sqrt(dif(minima, expected minima)^2+dif(maxima, expected maxima)^2))/N
    where minima is a vector containing the angles locations of all valleys,
    where maxima is a vector containing the angles locations of all peaks,
    and N is the total number of peaks ad valleys.
    Parameters: - observed: experimental data.
                - expected: values from fitted parameters.
                - angles: in degrees.
    Returns: resulting value from evaluation.
    '''
    # peaks and values for observation
    prominence = int(np.std(observed)) / 2
    observed_peaks = find_peaks(observed)[0]
    observed_valleys = find_peaks(np.max(observed) - observed, prominence=prominence)[0]
    # peaks and values for expected fit
    prominence = int(np.std(expected)) / 2
    expected_peaks = find_peaks(expected)[0]
    expected_valleys = find_peaks(np.max(expected) - expected, prominence=prominence)[0]

    # min number of peaks and valleys:
    n_peaks = np.min([len(expected_peaks), len(observed_peaks)])
    n_valleys = np.min([len(expected_valleys), len(observed_valleys)])

    # try:
    # check beginning of curves for half peak or valley
    if (observed_valleys[0] < observed_peaks[0]):
        first_valley_width = 2 * (angles[observed_peaks[0]] - angles[observed_valleys[0]])
        first_peak_width = angles[observed_valleys[1]] - angles[observed_valleys[0]]
    else:
        first_peak_width = 2 * (angles[observed_valleys[0]] - angles[observed_peaks[0]])
        first_valley_width = angles[observed_peaks[1]] - angles[observed_peaks[0]]

    # check end of curves for half peak or valley
    if (observed_valleys[n_valleys - 1] < observed_peaks[n_peaks - 1]):
        last_peak_width = 2 * (angles[observed_peaks[n_peaks - 1]] - angles[observed_valleys[n_valleys - 1]])
        last_valley_width = angles[observed_peaks[n_peaks - 1]] - angles[observed_peaks[n_peaks - 2]]
    else:
        last_valley_width = 2 * (angles[observed_valleys[n_valleys - 1]] - angles[observed_peaks[n_peaks - 1]])
        last_peak_width = angles[observed_valleys[n_valleys - 1]] - angles[observed_valleys[n_valleys - 2]]

    # calculate widths for the rest of the curve
    if (n_valleys < n_peaks):
        valleys_width = angles[observed_peaks[1: n_peaks]] - angles[observed_peaks[0: n_peaks - 1]]
    else:
        if (n_peaks == 2):
            valleys_width = np.concatenate([[first_valley_width], [angles[observed_peaks[1]] - angles[observed_peaks[0]]], [last_valley_width]])
        else:
            valleys_width = np.concatenate([[first_valley_width], angles[observed_peaks[2: n_peaks]] - angles[observed_peaks[1: n_peaks - 1]], [last_valley_width]])

    if (n_peaks < n_valleys):
        peaks_width = angles[observed_valleys[1: n_valleys]] - angles[observed_valleys[0: n_valleys - 1]]
    else:
        if (n_valleys == 2):
            peaks_width = np.concatenate([[first_peak_width], [angles[observed_valleys[1]] - angles[observed_valleys[0]]], [last_peak_width]])
        else:
            peaks_width = np.concatenate([[first_peak_width], angles[observed_valleys[2: n_valleys]] - angles[observed_valleys[1: n_valleys - 1]], [last_peak_width]])

    # auxiliary variables
    min_diffs_abs_aux = np.abs(angles[observed_valleys[:n_valleys]] - angles[expected_valleys[:n_valleys]])
    min_len = np.min([len(min_diffs_abs_aux), len(valleys_width)])
    min_diffs_abs = min_diffs_abs_aux[:min_len] / valleys_width[:min_len]

    max_diffs_abs_aux = np.abs(angles[observed_peaks[:n_peaks]] - angles[expected_peaks[:n_peaks]])
    max_len = np.min([len(max_diffs_abs_aux), len(peaks_width)])
    max_diffs_abs = max_diffs_abs_aux[:max_len] / peaks_width[:max_len]

    evaluation = (np.sum(min_diffs_abs) + np.sum(max_diffs_abs)) / (n_peaks + n_valleys)

    # print("Peak widths: ", peaks_width)
    # print("Valley widths: ", valleys_width)
    # print("Peak diffs", max_diffs_abs)
    # print("Valley diffs", min_diffs_abs)
    # print("mid valleys width: ", angles[observed_peaks[2: n_peaks]] - angles[observed_peaks[1: n_peaks - 1]])

    # except:
    #     return -1

    if (np.isnan(evaluation)):
        return -1

    if (verbose):
        print("-------------------------------------------")
        print("NELD: ", evaluation)
        print("Observed peaks: ", angles[observed_peaks])
        print("Expected peaks: ", angles[expected_peaks])
        print("Observed valleys: ", angles[observed_valleys])
        print("Expected valleys: ", angles[expected_valleys])
        print("-------------------------------------------\n")

    return evaluation

def chi_square(observed, expected, angles=None, ddoft=3, normalize=True):
    '''
    Calculates Chi^2 statistic.
    Parameters: - observed: experimental data.
                - expected: values from fitted parameters.
                - angles: dummy parameter.
                - ddof: default 3, degrees of freedom of the theorical equation.
                - normalize: option to normalize output by number of observations.
    Returns: - Chi^2
.             - Chi^2 / degrees_of_freedom.
    '''
    # degrees of freedom for the model
    df = len(observed) - 1 #- ddoft
    chi_aux = np.abs(np.square((observed - expected)) / expected)
    chi2 = np.sum(chi_aux[expected != 0])

    if normalize:
        return chi2 / df

    return  chi2

def chi_peak_combined(observed, expected, angles, ddoft=3, normalize=True):
    '''
    Calculates Chi^2 statistic times the peakloc_diff value.
    Parameters: - observed: experimental data.
                - expected: values from fitted parameters.
                - angles: in radians.
                - ddof: default 3, degrees of freedom of the theorical equation.
                - normalize: option to normalize output by number of observations.
    Returns: - Chi^2 statistic times the peakloc_diff value.
    '''
    return chi_square(observed, expected, ddoft=ddoft, normalize=normalize) * peakloc_diff(observed, expected, angles)

def rmse(observed, expected, angles=None, ddoft=3):
    '''
    Calculates the root mean square error.
    Parameters: - observed: experimental data.
                - expected: values from fitted parameters.
                - angles: dummy parameter.
                - ddof: default 3, degrees of freedom of the theorical equation.
                - statistic: test statistic to calculate RMSEA, default Chi^2.
    Returns: RMSEA value
    '''

    return np.sqrt(np.mean((observed-expected)**2))

def r2(observed, expected, angles=None, ddoft=3):
    '''
    Calculates the R² metric.
    Parameters: - observed: experimental data.
                - expected: values from fitted parameters.
                - angles: dummy parameter.
                - ddof: default 3, degrees of freedom of the theorical equation.
                - statistic: test statistic to calculate RMSEA, default Chi^2.
    Returns: RMSE value
    '''

    return np.abs(r2_score(observed, expected))

def rmsea(observed, expected, angles=None, ddoft=3, statistic=chi_square):
    '''
    Calculates the root mean square error of approximation (RMSEA)
    for the specified test statistic.
    Parameters: - observed: experimental data.
                - expected: values from fitted parameters.
                - angles: dummy parameter.
                - ddof: default 3, degrees of freedom of the theorical equation.
                - statistic: test statistic to calculate RMSEA, default Chi^2.
    Returns: RMSEA value
    '''
    # sample size
    n = len(observed)
    test_statistic = statistic(observed, expected)
    # degrees of freedom for the model
    df = n - ddoft
    test_statistic_mdf = test_statistic - df

    return np.sqrt(np.max([test_statistic_mdf / (df * (n - 1)), 0]))
    #return np.sqrt(np.abs(test_statistic_mdf / (df * (n - 1))))
    #return np.sqrt(np.max([(test_statistic / df - 1) / (n - 1), 0]))

def cfi(observed, expected, angles=None, ddoft=3, ddofb=3, inter_window=7, statistic=chi_square):
    '''
    Calculates the comparative fit index (CFI)
    for the specified test statistic.
    Parameters: - observed: experimental data.
                - expected: values from fitted parameters.
                - angles: dummy parameter.
                - ddoft: default 3, degrees of freedom of the of the theorical equation.
                - ddofb: default 3, rank of the polynomial for the interpolation model.
                - statistic: test statistic to calculate RMSEA, default Chi^2.
                - inter_window: default 7, interpolation window to interpolate obseved
                                data with a polynomial of order 3 using the function
                                scipy.signal.savgol_filter.
    Returns: RMSEA value
    '''
    # Calculate test statistic with observed and expected data
    test_statistic_target, _ = statistic(observed, expected)
    # Interpolation of observed data with polynomial of order 3
    interpolated = savgol_filter(observed, inter_window, ddofb)
    # Calculate test statistic with interpolated and expected data
    test_statistic_base, _ = statistic(observed, interpolated)
    # Auxiliary variables
    n = len(observed)
    # degrees of freedom for the target model
    dft =  n - ddoft
    # degrees of freedom for the interpolation model
    dfb = ddofb * (n - 1)
    statistic_target_mdft = test_statistic_target - dft
    statistic_target_mdfb = test_statistic_base - dfb
    #print(np.max([statistic_target_mddoft, 0]))
    #print(np.max([statistic_target_mddofb, 0]))
    #return 1 - np.max([statistic_target_mdft, 0]) / np.max([statistic_target_mdft, statistic_target_mdfb, 0])
    #return 1 - statistic_target_mdft / np.max([statistic_target_mdft, statistic_target_mdfb])
    return 1 - np.abs(statistic_target_mdft / statistic_target_mdfb)

def srmr_one_variable(observed, expected, angles=None):
    '''
    Calculates the standardised root mean square residual (SRMR)
    for a model with one observed varible.
    Parameters: - observed: experimental data.
                - expected: values from fitted parameters.
                - angles: dummy parameter.
    Returns: SRMR value
    '''
    # sample size
    n = len(observed)
    # observed variance
    observed_var = np.var(observed)
    # reproduced variance by the model
    reproduced_var = np.var(expected)

    var_diff = np.abs(observed_var - reproduced_var)
    std_mult = np.sqrt(np.abs(observed_var * reproduced_var))

    return np.sqrt((var_diff / std_mult))
