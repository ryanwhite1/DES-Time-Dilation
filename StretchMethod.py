# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 10:18:23 2023

@author: ryanw
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
from multiprocessing import Pool
import Methods
import pickle

plt.rcParams.update({"text.usetex": True})
plt.rcParams['font.family']='serif'
plt.rcParams['mathtext.fontset']='cm'

# will need the CIDs and zs later, so isolate them and make them arrays
CIDs = Methods.CIDs
Zs = Methods.Zs

# now define the bandpass info for the DES filters
DES_filters = {"g":[473, 150], "r":[642, 148], "i":[784, 147], "z":[926, 152], "y":[1009, 112]} # format: "band":[mean wavelength, bandwidth]
bandcolours = {"g": "tab:green", "r": "tab:orange", "i": "tab:red", "z": "tab:purple"} # some colours for plotting
z_center = DES_filters['z'][0]; z_width = DES_filters['z'][1]   # specifically get the z info 
width_frac = 1/16 # choose how much we're allowing the in rest-frame lightcurves either way of the mean. 1/2 would equate to the entire bandwidth, 1/4 is half of the bandwidth, etc

def stretch_chi_square(s, all_times, all_data, times, data, dataerr, domain):
    '''
    Parameters
    ----------
    s : float
        The estimated stretch parameter for which the times are inversely scaled by. I.e. s = 2 corresponds to the times being halved
    all_times : (1xN) np.array
        All of the time data for which to compare the individual light curve to
    all_data : (1xN) np.array
        All of the (scaled) flux data for which to compare the individual light curve to
    times : (1xN) np.array
        The timeseries array of the individual SN lightcurve
    data : (1xN) np.array
        The (scaled) flux data of the SN lightcurve
    dataerr : (1xN) np.array
        The (scaled) sigma errors of the flux data
    domain : float
        The allowed delta time either side of each data point to compare to the population of reference lightcurve data
    Returns
    -------
    chisq : float
        The calculated (reduced) chisq for the fit of stretch s given the reference data and the individual lightcurve data
    '''
    x = times / s   # de-stretch the time data by the estimated stretch amount
    rel_data = np.zeros(len(x)) # initialise array for the reference data
    for k in range(len(x)):
        # get the indices for points within +/- domain about the kth stretched time data
        args = np.argwhere((x[k] - domain <= all_times) & (all_times <= x[k] + domain)).flatten()
        
        # if len(args) == 0:  # if there are no points within that delta range, 
        #     rel_data[k] = 1e2   # set the y value at this point to be huge, so the chi square fit doesn't like this stretch value!
        # else:
        #     rel_data[k] = np.median(all_data[args])     # get the median value of the reference data in this domain
    # chisq = np.sum(np.square((data - rel_data) / dataerr)) / (len(x) - 1)   # calculate the reduced chi square for the data against the reference data
        
        # only count data that is within the bounds of our reference curve towards the chi-square fitting...
        if len(args) > 0:  # if there are points within that delta range, 
            rel_data[k] = np.median(all_data[args])     # get the median value of the reference data in this domain
    valid_args = np.argwhere(rel_data > 0).flatten()
    chisq = np.sum(np.square((data[valid_args] - rel_data[valid_args]) / dataerr[valid_args])) / (len(valid_args) - 1)   # calculate the reduced chi square for the data against the reference data
    
    return chisq


def binned_dispersions(xdata, ydata, dN, maximum=False):
    ''' Finds the median dispersion in some binned data. 
    Parameters
    ----------
    xdata : array
        x-values for points
    ydata : array
        y-values for points
    dN : int
        Number of bins to sort through
    maximum : bool
        If true, we return the maximum dispersion value across the light curve (for the purposes of an error floor)
    Returns
    -------
    dispersion : float
        The median dispersion across each of the bins
    sigma : float
        The standard deviation in these dispersions
    '''
    data_len = len(xdata) / dN  # find the number of points in each subdivision bin
    data_len = int(data_len + 1) if round(data_len) != data_len else int(data_len)  # handle int division
    
    # now we should sort the data to make the following for loop easier to code
    sort_inds = np.argsort(xdata)   # sort the data according to x-value
    sort_xdata = xdata[sort_inds]; sort_ydata = ydata[sort_inds]
    bin_sigmas = np.zeros(dN)
    
    for j in range(dN):     # for each subdivision
        lower_ind = j * data_len; upper_ind = (j + 1) * data_len    # get the index range of the SORTED data we want to look at
        if upper_ind > len(sort_xdata):  # need to handle the maximum upper bound index in case it tries to go over the maximum allowed index
            upper_ind = len(sort_xdata)
        vals = sort_ydata[lower_ind:upper_ind]    # get the ydata in the time range we want
        bin_sigmas[j] = np.std(vals)        # get the standard deviation of this data subdivision
    
    bin_sigmas = bin_sigmas[~np.isnan(bin_sigmas)]  # remove any nans from the array in case final bin has only 1 element inside
    if maximum:
        dispersion = max(bin_sigmas)
    else:
        dispersion = np.median(bin_sigmas)  # get the median standard deviation
    sigma = np.sqrt(sum([sigma**2 for sigma in bin_sigmas])) / dN  # get the standard deviation of standard deviations

    return dispersion, sigma

def reference_dispersion(cid, z, fit_band, bN=30, dN=30, method='linear'):
    ''' Finds the median dispersion of the reference lightcurve photometry for each value of some scaling variation parameter
    Parameters
    ----------
    cid : int
        The supernova CID
    z : float
        The estimated redshift of the supernova
    fit_band : str
        The filter of the *individual* light curves that we want to fit
    bN : int
        The number of stretch parameters to test
    dN : int
        The number of subdivisions of the data to look at. Generally more is better, but too many will mean we take the
        standard deviation of very few numbers
    method : str
        One of ['linear', 'power']. Linear case is scaling by 1/(1 + bz) for 0 <= b <= 2. Power case is scaling by 1/(1+z)^b for -1 <= b <= 2
    Returns
    -------
    bs : np.array
        Values of the scaling variation parameter that were looked at
    dispersions : np.array
        Median dispersions of the reference light curve subdivisions, for each value of the scaling variation parameter, b
    sigmas : np.array
        The standard deviation in the dispersions across the subdivisions for each b
    '''
    
    dispersions = np.zeros(bN)
    sigmas = np.zeros(bN)
    if method == 'linear':
        bs = np.linspace(0, 2, bN)
    else:   # power
        bs = np.linspace(-1, 2, bN)
    
    all_tdata, all_data, _, _, _, _, SNe_z, _ = Methods.get_reference_curve(cid, z, fit_band, scale=False, full_return=True)
    
    for i in range(bN):
        # now scale the time by the desired (1 + bz) or (1 + z)^b stretch
        if method == 'linear':
            tdata = all_tdata / (1 + bs[i] * SNe_z)
        else:   # must be power law
            tdata = all_tdata / (1 + SNe_z)**bs[i]
            
        dispersions[i], sigmas[i] = binned_dispersions(tdata, all_data, dN)  # get the median standard deviation and sigma error
    
    return bs, dispersions, sigmas
    
def get_stretch(cid, z, fit_band, require_prepeak=False):
    ''' Finds the stretch parameter of a supernova light curve to fit to a (dynamically determined) reference population
    Parameters
    ----------
    cid : int
        The supernova CID
    z : float
        The estimated redshift of the supernova
    fit_band : str
        The filter of the *individual* light curves that we want to fit
    require_prepeak : bool
        If true, only fit light curves that have data before the supposed peak. 
    Returns
    -------
    (width, sigma) tuple
        width : float
            The lightcurve width, where 1/width scales it to fit a standard reference population
        sigma : float
            The STD error in the width, found via 200 samples of monte carlo resampling
    '''
    min_sample = 5      # minimum number of photometry points that we want in our fitting
    
    # get the data for the SN we want to look at
    tsim, sim, _ = Methods.get_SN_sim(cid, fit_band)     # get SALT fit
    tdata, data, data_err = Methods.get_SN_data(cid, fit_band)   # get data
    if tsim.size == 0 or tdata.size < min_sample: # if no data (or too small a dataset), we'll just return (0, 0) and catch those later
        return (0, 0, 0, 0)
    if require_prepeak:     # if we require data before the peak, we'll similarly return (0, 0) if there is no negative time values
        prepeak_data = tdata[tdata <= 0]
        if len(prepeak_data) == 0:
            return (0, 0, 0, 0)
    data /= max(sim); data_err /= max(sim)  # normalise the data based on the SALT fit
    tdata = tdata.to_numpy(); data = data.to_numpy(); data_err = data_err.to_numpy() # convert those pandas dataframes into arrays
    
    all_tdata, all_data, bands = Methods.get_reference_curve(cid, z, fit_band) # get the reference curve for this SNe
    if len(all_tdata) <= 100:    # arbitrarily chosen (for the moment) minimum population for the reference photometry
        return (0, 0, 0, 0)
    
    # now get all of the target SNe data points that are within the lower and upper bounds (in time) of our reference curve
    use_indices = np.where((tdata <= max(all_tdata)) & (tdata >= min(all_tdata)))[0]
    
    if len(use_indices) < min_sample:   # if we don't have enough samples, ignore it
        return (0, 0, 0, 0)
    
    tdata = tdata[use_indices]; data = data[use_indices]; data_err = data_err[use_indices]
    
    # now to perform the monte carlo analysis on the data in reference to the reference photometry
    domain = 2      # the wiggle room (in days) each side of the data point to compare to the reference photometry
    iters = 200     # number of monte carlo iterations
    # sampled_data = np.zeros((len(tdata), iters))    # initialise the array for storing the resampled data
    # for j in range(len(tdata)):
    #     sampled_data[j, :] = np.random.normal(data[j], data_err[j], iters)  # normally sample each data point according to its error, store it in the array
    sampled_data = np.random.normal(data, data_err, (iters, len(tdata)))    # normally sample each data point according to its error, store it in the array
    sampled_widths = np.zeros(iters) # initialise array for the calculated widths
    # now to perform the monte carlo fitting
    for j in range(iters):
        # for each iteration, perform a chi square minimisation using the stretch_chi_square function, with an initial guess of s = (1 + z)
        result = opt.minimize(stretch_chi_square, [1 + z], 
                              args=(all_tdata, all_data, tdata, sampled_data[j, :], data_err, domain), 
                              method='Nelder-Mead') # seem to get best results with nelder-mead
        sampled_widths[j] = float(result.x)    # store the result in the array
    # now calculate what the stretch should be for the original data
    true = opt.minimize(stretch_chi_square, [1 + z], 
                          args=(all_tdata, all_data, tdata, data, data_err, domain), 
                          method='Nelder-Mead')
    width = float(true.x)
    data_err = np.std(sampled_widths)
    err_floor_prop, _ = binned_dispersions(all_tdata, all_data, 30)     # get the error floor proportion of the total width

    # err_floor = err_floor_prop * width  # get the absolute error floor
    err_floor = err_floor_prop * (1 + z)  # get the absolute error floor
    sigma = max(err_floor, data_err)   # error floor on the data based off of the dispersion in the reference photometry
    # print(f"ref_err={round(ref_err, 3)}, data_err={round(data_err, 3)}, total_err={sigma}")

    return [width, sigma, err_floor, data_err] # return the found stretch and the sigma error

def modified_linear(x, b):
    '''Modified linear model, with free gradient but enforced y intercept of y=1 with x=1=(1+z)'''
    return b * (x - 1) + 1
def power_1z(x, b):
    '''Power model, for x=(1 + z), with a free power parameter'''
    return x**b
def linear_model(x, m, c):
    '''Basic linear model with free gradient and intercept'''
    return m * x + c
def linear_chi_square(b, xdata, ydata, yerr):
    chisq = np.sum(np.square((power_1z(xdata, b) - ydata) / yerr)) / (len(xdata) - 1)
    return chisq

def single_widths(cids, zs, fit_band='z', require_prepeak=False):
    ''' Single core computation of lightcurve widths.
    Parameters
    ----------
    cids : np.array
        Array of all of the supernova CIDs to work through
    zs : np.array
        Array of all of the supernova redshifts, in the same order as the cids
    fit_band : str
        The filter of the *individual* light curves that we want to fit
    require_prepeak : bool
        If true, only fit light curves that have data before the supposed peak. 
    Returns
    -------
    cut_zs : np.array
        The quality cut redshifts corresponding to the quality cut found widths
    cut_widths : np.array
        Quality cut (to ensure no 0 values) widths
    cut_werr : np.array
        Quality cut (to ensure no 0 values) width errors
    '''
    widths = np.zeros((len(cids), 2)) # initialise array to store data
    for i, cid in enumerate(cids):  # iterate over each SN
        widths[i, :] = get_stretch(cid, zs[i], fit_band, require_prepeak)  # get width and err and store in array
                
    cut_zs = zs[np.where(widths[:, 0] != 0)]     # don't want 0 width points
    cut_cids = cids[np.where(widths[:, 0] != 0)]
    widths = widths[np.where(widths[:, 0] != 0)]
    
    use = np.where(widths[:, 1] != 0)   # don't want points with no uncertainty
    cut_cids = cut_cids[use] 
    cut_zs = cut_zs[use] 
    cut_widths = widths[:, 0][use] 
    cut_werr = widths[:, 1][use]
    cut_werr_floor = widths[:, 2][use]
    cut_werr_data = widths[:, 3][use]
    
    return cut_cids, cut_zs, cut_widths, cut_werr, cut_werr_floor, cut_werr_data       # zs, widths, width_err, width_err_floor, width_err_data

def multi_widths(cids, zs, fit_band='z', require_prepeak=False):
    ''' Multiprocessing computation of the lightcurve widths. Analogous to single core version (without plot functionality). 
    Parameters
    ----------
    cids : np.array
        Array of all of the supernova CIDs to work through
    zs : np.array
        Array of all of the supernova redshifts, in the same order as the cids
    fit_band : str
        The filter of the *individual* light curves that we want to fit
    require_prepeak : bool
        If true, only fit light curves that have data before the supposed peak. 
    Returns
    -------
    cut_zs : np.array
        The quality cut redshifts corresponding to the quality cut found widths
    cut_widths : np.array
        Quality cut (to ensure no 0 values) widths
    cut_werr : np.array
        Quality cut (to ensure no 0 values) width errors
    '''
    with Pool() as pool:    # initialise pool of workers
        widths = pool.starmap(get_stretch, [(cids[i], zs[i], fit_band, require_prepeak) for i in range(len(cids))])    # fill widths list with widths and errs
    widths = np.array(widths)   # convert above list into array so we can slice
    print(widths.shape)
    cut_zs = zs[np.where(widths[:, 0] != 0)]     # don't want 0 width points
    cut_cids = cids[np.where(widths[:, 0] != 0)]
    widths = widths[np.where(widths[:, 0] != 0)]
    
    use = np.where(widths[:, 1] != 0)   # don't want points with no uncertainty
    cut_cids = cut_cids[use] 
    cut_zs = cut_zs[use]
    cut_widths = widths[:, 0][use] 
    cut_werr = widths[:, 1][use]
    cut_werr_floor = widths[:, 2][use]
    cut_werr_data = widths[:, 3][use]
    
    return cut_cids, cut_zs, cut_widths, cut_werr, cut_werr_floor, cut_werr_data       # zs, widths, width_err, width_err_floor, width_err_data


def multi_scaling(cid, z, fitband, bN, dN, method):
    ''' Sub-function for use within `plot_disp_curve`, specifically for using multiprocessing. 
    Returns a tuple that is friendly for multiprocessing.
    Parameters
    ----------
    cid : int
        cid for the desired SN
    z : float
        redshift of the SN
    fit_band : str
        The filter of the *individual* light curves that we want to fit
    bN : int
        How many different values of the scaling parameter to look at
    dN : int
        Number of subdivisions of the entire reference data for which to look at dispersions
    method : str
        One of ['linear', 'power']. Linear case is scaling by 1/(1 + bz) for 0 <= b <= 2. Power case is scaling by 1/(1+z)^b for -1 <= b <= 2
    Returns
    -------
    (bool, np.array) tuple
        The bool says whether or not to use this value (if False, there are nans and this shouldnt be used!)
        The np.array contains the mean dispersions for each value of b
    '''
    _, dispersions, _ = reference_dispersion(cid, z, fitband, bN=bN, dN=dN, method=method)
    if True in np.isnan(dispersions):
        return (False, dispersions)
    else:
        return (True, dispersions)
    
def gen_disp_curve(cid, z, fitband, bN=30, dN=30, method='power', single=False):
    ''' Calculates the mean dispersions per value of some scaling parameter according to the scaling function. 
    Parameters
    ----------
    cid : int or np.array
        list of cids for the desired SNs
    z : float or np.array
        list of redshifts of the SNs
    fit_band : str
        The filter of the *individual* light curves that we want to fit
    bN : int
        How many different values of the scaling parameter to look at
    dN : int
        Number of subdivisions of the entire reference data for which to look at dispersions
    method : str
        One of ['linear', 'power']. Linear case is scaling by 1/(1 + bz) for 0 <= b <= 2. Power case is scaling by 1/(1+z)^b for -1 <= b <= 2
    single : bool
        If true, uses the single core version of the scaling function. If False, defaults to multiprocessing version. 
    Returns
    -------
    disps : (~len(cid) x bN) np.array
        the array of dispersions for each cid/scaling value to plot later
    '''
    disps = np.zeros((len(cid), bN))    # initialise array to store dispersions across the multiple SNs
    if single: # single core version
        use = np.ones(len(cid), dtype=bool) # initialise filtering array. Assume we want to use all iterations, so we have N True vals
        for i in range(len(cid)):   # now iterate over the SN
            bs, dispersions, sigmas = reference_dispersion(cid[i], z[i], fitband, bN=bN, dN=dN, method=method)
            if True in np.isnan(dispersions):   # check if the dispersions have a nan value
                use[i] = False  # we don't want to use this iteration, filter it out later
            else:
                disps[i, :] = dispersions
        disps = disps[use]  # filter out iterations with nans
    else: # multi core
        with Pool() as pool:
            disps = pool.starmap(multi_scaling, [(cid[i], z[i], fitband, bN, dN, method) for i in range(len(cid))])
        # now we want to unpack the multiprocessing output
        use = np.array([x for (x, y) in disps])     # array of whether we want to use the iteration or not
        disps = np.array([y for (x, y) in disps])   # array of arrays for our dispersions
        disps = disps[use]
    
    return disps    



def reference_pop(cid, z, band, width):
    ''' Finds the population of a reference curve given the inputs.
    Parameters
    ----------
    cid : int or np.array
        cid of the SNe
    z : float
        redshifts of the SNe
    band : str
        The filter of the *individual* light curves that we want to fit
    width : float
        The fraction of bandwidth to sample from either side of the mean wavelength
    Returns
    -------
    int
        Number of data points within a reference curve
    '''
    all_tdata, _, _ = Methods.get_reference_curve(cid, z, band, width_frac=width)
    return len(all_tdata)
def calculate_reference_pops(cids, zs, width_iterable, fit_band):
    ''' Will calculate the reference population for many SNe and many width fractions. Multiprocessing functionality!
    Parameters
    ----------
    cids : np.array
        Array of all of the supernova CIDs to work through
    zs : np.array
        Array of all of the supernova redshifts, in the same order as the cids
    width_iterable : list/np.array
        An array of widths to iterate over to find the population given the width param in the reference_pop function.
    fit_band : str
        The filter of the *individual* light curves that we want to fit
    Returns
    -------
    cids : np.array
        Same as input
    zs : np.array
        Same as input
    width_iterable : np.array
        Same as input
    reference_pops : MxN np.array
        for M cids/zs, and N width iterables, this is the population of the reference population for each parameter combination
    '''
    # 
    reference_pops = np.zeros((len(cids), len(width_iterable)))
    for j, width in enumerate(width_iterable):
        with Pool() as pool:
            pops = pool.starmap(reference_pop, [(cids[i], zs[i], fit_band, width) for i in range(len(cids))])
        reference_pops[:, j] = pops
    ## below is a single core version of the above in case needed later
    # for i, cid in enumerate(cids):
    #     z = zs[i]
    #     for j, width in enumerate(width_iterable):
    #         all_tdata, _, _ = Methods.get_reference_curve(cid, z, 'i')
    #         reference_pops[i, j] = len(all_tdata)
            
    return cids, zs, width_iterable, reference_pops
        


def main():
    ''' Objective here is to generate and store the data. 
    '''
    print("Starting data processing...")
    
    ## below code looks at the reference populations vs the width factor
    widths = np.array([1/2, 1/4, 1/8, 1/16, 1/32])
    for band in ['g', 'r', 'i', 'z']:
        cids, zs, width_iterable, reference_pops = calculate_reference_pops(CIDs, Zs, widths, band)
        with open(f'reference_pops_{band}band', 'wb') as file:
            pickle.dump([cids, zs, width_iterable, reference_pops], file)
        print(f"Finished {band} reference pops.")
    print("Finished counting reference populations.")
    
    
    ### the below code is looking at the median dispersion vs scaling parameter
    
    # # the below generate the dispersion curves of the light curves
    for band in ['g', 'r', 'i', 'z']:
        disps = gen_disp_curve(CIDs, Zs, band, bN=30, dN=15, method='power')
        with open(f'power_dispersion_pickle_{band}', 'wb') as file:
            pickle.dump(disps, file)
        print(f"Finished {band} disp. curves.")
    print("Finished generating dispersion curves in power law case. ")
    # disps = gen_disp_curve(CIDs, Zs, 'z', bN=30, dN=15, method='linear')
    # with open('linear_dispersion_pickle', 'wb') as file:
    #     pickle.dump(disps, file)
    # print("Finished generating dispersion curves in linear case. ")

    ## the below code calculates the lightcurve widths and pickles them to use later
    
    # start with z-band data
    stop = len(Zs)
    use_cids = CIDs[:stop]
    use_zs = Zs[:stop]
    # zs, widths, werr = single_widths(use_cids, Zs[:stop])
    
    for band in ['g', 'r', 'i', 'z']:
        cids, zs, widths, werr, werr_floor, werr_data = multi_widths(use_cids, use_zs, fit_band=band, require_prepeak=False)
        dataframe = {"CID":cids, "z":zs, "Width":widths, "Width_err":werr, 'Width_err_floor':werr_floor, 'Width_err_data':werr_data}
        dataframe = pd.DataFrame(dataframe)
        with open(f'{band}BAND_pickled_data', 'wb') as file:
            pickle.dump(dataframe, file)
    print("Finished calculating widths!")
    
    # use the below code to re-run with the requirement of pre-peak points!
    for band in ['g', 'r', 'i', 'z']:
        cids, zs, widths, werr, werr_floor, werr_data = multi_widths(use_cids, use_zs, fit_band=band, require_prepeak=True)
        dataframe = {"CID":cids, "z":zs, "Width":widths, "Width_err":werr, 'Width_err_floor':werr_floor, 'Width_err_data':werr_data}
        dataframe = pd.DataFrame(dataframe)
        with open(f'{band}BAND_pickled_data_PREPEAK', 'wb') as file:
            pickle.dump(dataframe, file)
    print("Finished calculating widths (requiring pre-peak data)!")



if __name__ == "__main__":
    main()
    
    