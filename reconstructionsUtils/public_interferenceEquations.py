import numpy as np
import pandas as pd
import math

def silica_refractive_index(wl):
    '''
    Calculates refractive index for SiO2 according to the dispersion
    formula in https://refractiveindex.info/?shelf=main&book=SiO2&page=Malitson
    Parameter:
        wl: wavelength in nm.
    Returns: SiO2 refractive index for wl.
    '''
    wl = wl / 1000 # convert to µm
    aux1 = 0.6961663 * np.square(wl) / (np.square(wl) - 0.0684043**2)
    aux2 = 0.4079426 * np.square(wl) / (np.square(wl) - 0.1162414**2)
    aux3 = 0.8974794 * np.square(wl) / (np.square(wl) - 9.896161**2)

    return np.sqrt(aux1 + aux2 + aux3 + 1)

def silicon_refractive_index(wl):
    '''
    Estimates refractive index for Si according to data table from:
    https://www.filmetrics.com/refractive-index-database/Si/Silicon
    Parameter:
        wl: wavelength in nm
    Returns: Si refractive index for wl
    '''
    # load data file with wavelengths and refractive indexes information
    data = pd.read_csv('./reconstructionsUtils/auxiliaryFiles/Si_from_filmetrics.txt', delimiter="\t")
    data.drop(['k'], axis=1, inplace=True)
    data.columns = ['wl', 'n']

    if len(data[data.wl==wl]) == 1:
        # wl in data file
        return data.n[data.wl==wl].values[0]
    else:
        # add column 'difference' to data, substracts wl to data wavelengths
        data = data.assign(difference = wl - data.wl.values)
        # finds closests wavelngths to wl
        wl_min, n_min, _ = data[data.difference > 0].iloc[-1]
        wl_max, n_max, _ = data[data.difference < 0].iloc[0]
        # Linear interpolation taking x=wl and y=n
        return (n_max - n_min) / (wl_max - wl_min) * (wl - wl_min) + n_min

def h2o_refractive_index(wl):
    '''
    Estimates refractive index for water according to data table from:
    https://refractiveindex.info/?shelf=main&book=H2O&page=Segelstein
    Parameter:
        wl: wavelength in nm
    Returns: Si refractive index for wl
    '''
    # load data file with wavelengths and refractive indexes information
    data = pd.read_csv('./reconstructionsUtils/auxiliaryFiles/h2o_Hale.csv')
    data.drop(['k'], axis=1, inplace=True)
    data.columns = ['wl', 'n']

    # wavelenght to microns
    wl = wl / 1000
    if len(data[data.wl==wl]) == 1:
        # wl in data file
        return data.n[data.wl==wl].values[0]
    else:
        # add column 'difference' to data, substracts wl to data wavelengths
        data = data.assign(difference = wl - data.wl.values)
        # finds closests wavelngths to wl
        wl_min, n_min, _ = data[data.difference > 0].iloc[-1]
        wl_max, n_max, _ = data[data.difference < 0].iloc[0]
        return (n_max - n_min) / (wl_max - wl_min) * (wl - wl_min) + n_min

def t_media_ox(n, n_ox, theta):
    '''
    TE Fresnel transmission coefficent from sample/media to SiO2 layer.
    Parameters:
        n: refractive index of sample or immersion media.
        n_ox: refractive index SiO2 layer.
        theta: beam's incidence angle from media to SiO2 layer in rad.
        Returns: TE transmission Fresnel coefficient.
    '''

    nox_cos_theta_ox = n_ox * np.sqrt(1 - np.square(n / n_ox * np.sin(theta)))
    n_cos_theta = n * np.cos(theta)


    return 2 * n_cos_theta / (n_cos_theta + nox_cos_theta_ox)

def r_media_ox(n, n_ox, theta):
    '''
    TE Fresnel reflection coefficent between sample/media and SiO2 layer.
    Parameters:
        n: refractive index of sample or immersion media.
        n_ox: refractive index SiO2 layer.
        theta: beam's incidence angle from media to SiO2 layer in rad.
        Returns: TE transmission Fresnel coefficient.
    '''

    nox_cos_theta_ox = n_ox * np.sqrt(1 - np.square(n / n_ox * np.sin(theta)))
    n_cos_theta = n * np.cos(theta)

    return (n_cos_theta - nox_cos_theta_ox) / (n_cos_theta + nox_cos_theta_ox)

def t_ox_media(n_ox, n, theta):
    '''
    TE Fresnel transmission coefficent from SiO2 layer to sample/media.
    Parameters:
        n: refractive index of sample or immersion media.
        n_ox: refractive index SiO2 layer.
        theta: beam's incidence angle from media to SiO2 layer in rad.
    Returns: TE transmission Fresnel coefficient.
    '''
    nox_cos_theta_ox = n_ox * np.sqrt(1 - np.square(n / n_ox * np.sin(theta)))

    return 2 * nox_cos_theta_ox / (n * np.cos(theta) + nox_cos_theta_ox)

def r_ox_si(n, n_ox, n_si, theta):
    '''
    TE Fresnel reflection coefficent from Si layer inside SiO2 layer.
    Parameters:
        n: refractive index of sample or immersion media.
        n_ox: refractive index SiO2 layer.
        n_si: refractive index Si layer.
        theta: beam's incidence angle from media to SiO2 layer in rad.
    Returns: TE total reflection Fresnel coefficient.
    '''
    nox_cos_theta_ox = n_ox * np.sqrt(1 - np.square(n / n_ox * np.sin(theta)))

    nsi_cos_theta_si = n_si * np.sqrt(1 - np.square(n / n_si * np.sin(theta)))

    return (nox_cos_theta_ox - nsi_cos_theta_si)/ (nox_cos_theta_ox + nsi_cos_theta_si)

def rte(n, n_ox, n_si, theta, wl, d_ox):
    '''
    TE Fresnel reflection coefficent from Weaver paper. The expression is taken from
    Max and Wolf 1.6.2-1.6.3 and Lambacher 1996.
    Parameters:
        n: refractive index of sample or immersion media.
        n_ox: refractive index SiO2 layer.
        n_si: refractive index silicon layer.
        wl: beam's wavelenght in nm.
        d_ox: depth of SiO2 layer.
        Returns: TE reflection Fresnel coefficient.
    '''
    delta = 4 * np.pi * n_ox * d_ox * np.sqrt(1 - np.square(n / n_ox * np.sin(theta))) / wl
    r_mox = r_media_ox(n, n_ox, theta)
    r_oxsi_exp_idelta = r_ox_si(n, n_ox, n_si, theta) * np.exp(1j * delta)

    return (r_mox + r_oxsi_exp_idelta) / (1 + r_mox * r_oxsi_exp_idelta)

def phase(n, theta, wl, h):
    '''
    Phase shift experienced by reflected beam compared to a direct beam at a distance h
    from the SiO2 layer's surface.
    Parameters:
        n: refractive index of sample or immersion media.
        theta: beam's incidence angle from media to SiO2 layer in rad.
        wl: beam's wavelenght in nm.
        h: distance from SiO2 layer's surface.
    Returns: phase shift.
    '''
    return 4 * np.pi / wl * n * h * np.cos(theta)

def one_beam_intensity(n, n_ox, n_si, theta, wl, d_ox, h):
    '''
    Interference intensity at a distance h from the SiO2 layer's surface when
    the incident beam interferes with its reflection.
    Parameters:
        n: refractive index of sample or immersion media.
        n_ox: refractive index SiO2 layer.
        n_si: refractive index silicon layer.
        theta: beam's incidence angle from media to SiO2 layer in rad.
        wl: beam's wavelenght in nm.
        d_ox: depth of SiO2 layer.
        h: distance from SiO2 layer's surface.
    Returns: intensity in a.u.
    '''
    phase_diff = phase(n, theta, wl, h)
    # TE Fresnel coefficient from Born & Wolf book and same as papers
    r = rte(n, n_ox, n_si, theta, wl, d_ox)

    return np.square(np.absolute(1 + r * np.exp(1j * phase_diff)))

def one_beam_intensity_expanded_expression(n, n_ox, n_si, theta, wl, d_ox, h):
    '''
    Interference intensity at a distance h from the SiO2 layer's surface when
    the incident beam interferes with its reflection.
    Parameters:
        n: refractive index of sample or immersion media.
        n_ox: refractive index SiO2 layer.
        n_si: refractive index silicon layer.
        theta: beam's incidence angle from media to SiO2 layer in rad.
        wl: beam's wavelenght in nm.
        d_ox: depth of SiO2 layer.
        h: distance from SiO2 layer's surface.
    Returns: intensity in a.u.
    '''
    phase_diff = phase(n, theta, wl, h)
    cos_phase_diff = np.cos(phase_diff)

    delta = 4 * np.pi * n_ox * d_ox * np.sqrt(1 - np.square(n / n_ox * np.sin(theta))) / wl
    cos_delta = np.cos(delta)

    r1 = r_media_ox(n, n_ox, theta)
    r2 = r_ox_si(n, n_ox, n_si, theta)
    r1r2 = r1 * r2

    C = 1 + r1r2**2 + 2 * r1r2 * cos_delta

    reff_square_mag = (r1**2 + r2**2 + 2 * r1r2 * cos_delta) / C

    reffexpphiplusconj = 2 / C * (r1 * cos_phase_diff * (1 + r2**2) + r1 * r1r2 * np.cos(phase_diff - delta) + r2 * np.cos(phase_diff + delta))

    return 1 + reff_square_mag + reffexpphiplusconj
