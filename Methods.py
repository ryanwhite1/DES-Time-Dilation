# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 15:16:39 2023

@author: ryanw
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import interpolate

# import the data, and perform a quality cut based on the SALT3 fit
sn = pd.read_csv('FITOPT000.FITRES', delim_whitespace=True, comment='#')
# sn = sn[sn["FITPROB"] > 0.6]
lc = pd.read_csv('FITOPT000.LCPLOT', delim_whitespace=True, comment='#')
prob1a = pd.read_csv('hubble_diagram_wPIa.txt', delim_whitespace=True, comment='#')

# now clean up the prob1a array
prob1a = prob1a.head(1634)  # truncate the data at this row, since all rows after don't use the same CID format
prob1a["CID"] = pd.to_numeric(prob1a["CID"]); prob1a["PROBIa_BEAMS"] = pd.to_numeric(prob1a["PROBIa_BEAMS"])    # convert array of strings to numbers
# now we filter both arrays based on their mutual SNe, and then by the PROBIa > 0.5 condition
commonCIDs = np.intersect1d(sn["CID"], prob1a["CID"], assume_unique=True)  # get common CIDs
prob1a = prob1a[np.isin(prob1a["CID"], commonCIDs)]
prob1a = prob1a[prob1a["PROBIa_BEAMS"] > 0.5]
sn = sn[np.isin(sn["CID"], prob1a["CID"])]

lc_uncleaned = lc
big_err_inds = lc[lc.FLUXCAL_ERR > 20].index
data_inds = lc[lc.DATAFLAG == 1].index
lc = lc.drop(np.intersect1d(big_err_inds, data_inds))   # this is causing an error in the scaling
# will need the CIDs and zs later, so isolate them and make them arrays
CIDs = sn["CID"].to_numpy()
Zs = sn["zHEL"].to_numpy()


# now define the bandpass info for the DES filters
DES_filters = {"g":[473, 150], "r":[642, 148], "i":[784, 147], "z":[926, 152], "y":[1009, 112]} # format: "band":[mean wavelength, bandwidth]

def get_SN_data(cid, band):
    ''' Gets the times, flux, and flux errors for a SN with cid in the {g, r, i, z} band. 
    Parameters
    ----------
    cid : int
        The CID of the SNe
    band : str
        One of {g, r, i, z} to say which band data to return.
    Returns
    -------
    t : pd.DataFrame
        Array of times of the data observed
    y : pd.DataFrame
        The flux of the data observed
    yerr : pd.DataFrame
        The error in the flux observed
    '''
    SN = lc[lc["CID"] == cid]   # restrict the SN dataframe to this cid
    SN = SN[SN["BAND"] == band]     # restrict the cid dataframe to this band
    SNdata = SN[SN["DATAFLAG"] == 1]    # only get the data (which has dataflag 1)
    t = SNdata["MJD"] - sn[sn["CID"] == cid]["PKMJD"].to_numpy()    # get the times relative to the peak day
    return t, SNdata["FLUXCAL"], SNdata["FLUXCAL_ERR"]

def get_SN_sim(cid, band):
    ''' Gets the times, flux, and flux errors for a SN SALT fit with cid in the {g, r, i, z} band. 
    Parameters
    ----------
    cid : int
        The CID of the SNe
    band : str
        One of {g, r, i, z} to say which band SALT fit to return.
    Returns
    -------
    t : pd.DataFrame
        Array of times of the model fit
    y : pd.DataFrame
        The expected flux in the model fit
    yerr : pd.DataFrame
        The error in the expected flux
    '''
    SN = lc[lc["CID"] == cid]   # restrict the SN datafram to this cid
    SN = SN[SN["BAND"] == band]     # restrict the cid dataframe to this band
    SNsim = SN[SN["DATAFLAG"] == 0]   # only get the SALT fit (which has dataflag 0)
    t = SNsim["MJD"] - sn[sn["CID"] == cid]["PKMJD"].to_numpy()    # get the times relative to the peak day
    return t, SNsim["FLUXCAL"], SNsim["FLUXCAL_ERR"]



def get_reference_curve(cid, z, fit_band, width_frac=1/16, scale=True, scale_factor=1., full_return=False):
    ''' Samples and generates a reference curve for a given SNe cid and redshift. 
    Parameters
    ----------
    cid : int
        The SNe CID
    z : int
        The redshift of the supernova
    fit_band : str
        The filter of the *individual* light curves that we want to fit
    width_frac : float
        The fraction of bandwidth to sample from either side of the mean wavelength
    scale : bool
        If True, will correct the reference curve for 1+z time dilation
    scale_factor : float
        How much to scale the reference curve for in time according to (1 + z)^b relation. Default of b=1 corresponds to 
        a 1/(1+z) scaling.
    full_return : bool
        If True, will return more information about the reference. See return cases.
    Returns
    -------
    all_tdata, all_data, bands : np.arrays
        If full_return==False, will return 3 arrays, each for the time data, and (normalised) flux data, and sampled bands for 
        the reference curve. The sampled bands are in a format that there is one entry in the array for each SNe from each band.
        i.e. if there were 4 SNe each sampled in 4 bands, there would be 16 entries.
    all_tdata, all_data, bands, all_data_err, valid_cids, valid_zs, SNe_z, running_index : np.arrays
        If full_return==True, will return 8 arrays. The first 3 are the same as in the full_return==False case. 
        all_data_err are the (normalised) error bars for each data point in all_data
        valid_cids gives the cids used in the reference curve. There may be duplicates (although extremely unlikely), as there is one entry for each band that an SNe is sampled from.
        valid_zs is as above, but the redshifts for each of the SNe sampled. (May be duplicates, but very unlikely)
        SNe_z is an array of the redshifts of each sampled SNe, but with one entry for each data point sampled. i.e. a z=0.4 SNe with 4 sampled points would have [0.4, 0.4, 0.4, 0.4]
        running_index is an array that has the number of data points of each sampled SNe
    '''
    index = np.argwhere(CIDs == cid)    # get the index of our supernova so that it doesn't get added to the reference curve
    # initialise some lists to add data to
    valid_cids = []
    valid_zs = []
    bands = []
    args_list = []
    if full_return:
        SNe_z = []
        running_index = []
    fit_band_center, fit_band_width = DES_filters[fit_band]
    wiggle = width_frac * fit_band_width # the wiggle room either side of the mean wavelength of our band
    # now we want to loop through each band and look at all of the relevant SN lightcurves we can add to our reference population
    for band in ['g', 'r', 'i', 'z']:
        # get the band center and bandwidth of the current band
        band_center = DES_filters[band][0]
        
        # we want to now find a range of redshifts that we can take in *this* band, to compare to our observed data in the z-band
        # for the SN with cid[i]. 
        # if we take the (z-band mean wavelength +/- wiggle room), and divide by the central wavelength of the *current* band,
        # we'll get upper and lower bounds on the redshift range of applicable SN lightcurves to add to our reference. 
        # These will usually be bigger than the mean wavelength of the current band, so we subtract 1 so that we 
        # are getting data from bands with lower mean wavelengths
        # i.e., for a high redshift SN that we're looking at in the z-band, subtracting 1 will mean we're looking at reference
        # SN in g/r/i bands at lower redshifts
        
        upper, lower = (band_center * (1 + z) + wiggle) / fit_band_center - 1, (band_center * (1 + z) - wiggle) / fit_band_center - 1
        args = np.argwhere((upper >= Zs) & (lower <= Zs)).flatten()
        if index in args:
            args = np.delete(args, np.argwhere(args == index).flatten())  # need to delete the FITTING SNe from the reference set to avoid circularity
        valid_cids.extend(CIDs[args]) # add those CIDs to the list of applicable references
        valid_zs.extend(Zs[args]) # add the redshifts of the reference SN to the list of applicable references
        bands += [band] * len(args) # add which band is applicable so we know which CID corresponds to which band
        args_list.append(len(args))
        
    del_inds = []
    all_tdata, all_data, all_data_err = [], [], [] # initialise lists for reference photometry
    # now to go through all of the reference photometry and normalise it, and add it to the arrays of all data
    for j, vcid in enumerate(valid_cids): 
        tsim, sim, _ = get_SN_sim(vcid, bands[j]) # get reference SN SALT fit
        tdata, data, data_err = get_SN_data(vcid, bands[j])    # get reference SN raw data
        if tsim.size == 0 or tdata.size == 0:
            del_inds.append(j)
            continue    # if no data here, just continue on with the loop and don't add this SN to the reference
        data /= max(sim)                 # normalise the flux based on the SALT fit
        data_err /= max(sim)             # as above but for error too
        if scale:
            tdata /= (1. + valid_zs[j])**scale_factor       # and scale the time by the expected (1+z) stretch
        all_tdata.extend(tdata); all_data.extend(data); all_data_err.extend(data_err) # add this data to the reference arrays!
        if full_return:
            SNe_z.extend([valid_zs[j]] * len(data))
            running_index.append(len(data))
            
    if len(del_inds) > 0:       # now get rid of lightcurves that didnt make the data cut
        valid_cids = np.delete(valid_cids, del_inds)
        valid_zs = np.delete(valid_zs, del_inds)
        bands = np.delete(bands, del_inds)
    
    all_tdata, all_data, all_data_err = np.array(all_tdata), np.array(all_data), np.array(all_data_err) # convert the reference lists into arrays
    
    if full_return:
        return all_tdata, all_data, bands, all_data_err, np.array(valid_cids), np.array(valid_zs), np.array(SNe_z), np.array(running_index)
    return all_tdata, all_data, bands
        
def bin_data(x, y, xerr=[], yerr=[], n=50):
    ''' Takes data and returns bins of the medians in the data (including binned errors). Error is calculated based on 
        gaussian uncertainty propagation formulae.
    Parameters
    ----------
    x : np.array
        X values for each data point
    y : np.array
        Y values for each data point (i.e. must be same length as x)
    xerr : list/np.array
        Errors in the x values. If non-empty, must have same length as x
    yerr : list/np,array
        Errors in the y values. If non-empty, must have same length as y
    n : int
        Number of data points to group together in each bin. 
    Returns
    -------
    xbins : np.array
        Array of median values within grouped bins
    ybins : np.array
        ""
    xerrbins (if xerr != []): np.array
        X axis error in each value within xbins
    yerrbins (if yerr != []): np.array
        Y axis error in each value within ybins
    '''
    # start by sorting the data so that we can easily group points within population bins
    p = x.argsort()     # obtain sorted arguments
    X = x[p]; Y = y[p]  # sort each array
    
    numbins = int(np.ceil(len(x) / n))  # calculate number of bins we'll need for the whole data having n points in each bin
    xbins = np.zeros(numbins)   # initialise arrays to store medians
    ybins = np.zeros(numbins)
    
    # now check each x/y error to see if non-empty, and sort/initialise arrays if so
    if len(xerr) != 0:
        Xerr = xerr[p]
        xerrbins = np.zeros(numbins)
    if len(yerr) != 0:
        Yerr = yerr[p]
        yerrbins = np.zeros(numbins)
    
    j = 0
    # now loop over data to put data in the bins
    for i in range(len(x)):
        I = i % n   # check if we're at the end of a bin. 
        if (I == 0 and i != 0) or (i == len(x) - 1): # if I == 0, then we need to calculate the median over the last n data points
            xbins[j] = np.median(X[j * n:i])    # calculate medians over last ~n data points
            ybins[j] = np.median(Y[j * n:i])
            # now use gaussian error propagation to calculate error in the last ~n data points
            if len(xerr) != 0:
                xerrbins[j] = np.sqrt(sum([sigma**2 for sigma in Xerr[j * n:i]])) / len(Xerr[j * n:i])
            if len(yerr) != 0:
                yerrbins[j] = np.sqrt(sum([sigma**2 for sigma in Yerr[j * n:i]])) / len(Yerr[j * n:i])
            j += 1 
    
    # now return the correct configuration of data given the initial inputs
    if len(xerr) != 0:
        if len(yerr) != 0:
            return xbins, ybins, xerrbins, yerrbins
        else:
            return xbins, ybins, xerrbins
    elif len(yerr) != 0:
        return xbins, ybins, yerrbins
    return xbins, ybins



def get_like_widths(data1, data2, widthcut=True, errorcut=True):
    ''' Takes in all of the width data from two bands, and returns the (unpacked) sorted data from SNe that are shared between them
    Parameters
    ----------
    data1 : pd.Dataframe
        Dataframe from the first band with columns {"CID", "z", "Width", "Width_err"} in any order
    data2 : pd.Dataframe
        Dataframe from the second band with columns {"CID", "z", "Width", "Width_err"} in any order
    widthcut : bool
        If True, only looks at width points that have a width of less than 4
    errorcut : bool
        If True, cuts out data points where the error is larger than the width signal
    Returns
    -------
    cids1 : np.array
        All of the cids in the first band from SNe that are shared amongst each of the two input datasets
    zs1 : np.array
        All of the redshifts in the first band from SNe that are shared amongst each of the two input datasets
    widths1 : np.array
        All of the widths in the first band from SNe that are shared amongst each of the two input datasets
    werr1 : np.array
        All of the width errors in the first band from SNe that are shared amongst each of the two input datasets
    cids2, zs2, widths2, werr2 : np.arrays
        As above, but for the second bands data
    '''
    # start by unpacking the data given
    cids1, cids2 = data1["CID"].to_numpy(), data2["CID"].to_numpy()
    zs1, zs2 = data1["z"].to_numpy(), data2["z"].to_numpy()
    widths1, widths2 = data1["Width"].to_numpy(), data2["Width"].to_numpy()
    werr1, werr2 = data1["Width_err"].to_numpy(), data2["Width_err"].to_numpy()
    
    # sorting the points according to cid will make the code easier for later
    # need to sort by cid, since it's the only unique identifier of SNe
    p1 = np.argsort(cids1); p2 = np.argsort(cids2)
    cids1, zs1, widths1, werr1 = cids1[p1], zs1[p1], widths1[p1], werr1[p1]
    cids2, zs2, widths2, werr2 = cids2[p2], zs2[p2], widths2[p2], werr2[p2]
    
    if errorcut:    # only care about values that have error less than the signal
        err_inds1 = np.less(werr1, widths1)     # find the points in the first band that have error < signal
        cids1, zs1, widths1, werr1 = cids1[err_inds1], zs1[err_inds1], widths1[err_inds1], werr1[err_inds1]  # only take the points from first band that have error < signal
        err_inds2 = np.less(werr2, widths2)     # find the points in the second band that have error < signal
        cids2, zs2, widths2, werr2 = cids2[err_inds2], zs2[err_inds2], widths2[err_inds2], werr2[err_inds2]  # as above, but in the second band
    
    if widthcut:    # perform a quality cut based on a maximum width
        wid_inds1 = np.less(widths1, 4)
        cids1, zs1, widths1, werr1 = cids1[wid_inds1], zs1[wid_inds1], widths1[wid_inds1], werr1[wid_inds1]
        wid_inds2 = np.less(widths2, 4)
        cids2, zs2, widths2, werr2 = cids2[wid_inds2], zs2[wid_inds2], widths2[wid_inds2], werr2[wid_inds2]
    
    # find the intersection of the two CID arrays (i.e. common points across the curves)
    _, data1_inds, data2_inds = np.intersect1d(cids1, cids2, return_indices=True)
    # print(data1_inds, data2_inds)
    # print(len(cids1), len(zs1), len(data1_inds), len(cids2), len(zs2), len(data2_inds))
    cids1, cids2 = cids1[data1_inds], cids2[data2_inds]
    zs1, zs2 = zs1[data1_inds], zs2[data2_inds]
    widths1, widths2 = widths1[data1_inds], widths2[data2_inds]
    werr1, werr2 = werr1[data1_inds], werr2[data2_inds]
    
    return cids1, zs1, widths1, werr1, cids2, zs2, widths2, werr2
    
    
    
    
    
        