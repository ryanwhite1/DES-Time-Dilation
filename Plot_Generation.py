# -*- coding: utf-8 -*-
"""
Various functions and code for plotting all of the data generated in StretchMethod.py. 


@author: ryanw
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib.legend_handler import HandlerTuple
import scipy.optimize as opt
import pickle
import pandas as pd
import Methods
import StretchMethod

def modified_linear(x, b):
    '''Modified linear model, with free gradient but enforced y intercept of y=1 with x=1=(1+z)'''
    return b * (x - 1) + 1
def power_1z(x, b):
    '''Power model, for x=(1 + z), with a free power parameter'''
    return x**b
def linear_model(x, m, c):
    '''Basic linear model with free gradient and intercept'''
    return m * x + c

# set LaTeX font for our figures
plt.rcParams.update({"text.usetex": True})
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'cm'

# import the CIDs and Zs from the methods file for our use in some plots
CIDs = Methods.CIDs
Zs = Methods.Zs
mB = Methods.sn['mB'].to_numpy()

# generally useful data for the plots later on
filenames = {'z': 'zBAND', 'i': "iBAND", 'r': 'rBAND', 'g': "gBAND"}    # filter filename formats
DES_filters = {"g": [473, 150], "r": [642, 148], "i": [784, 147], "z": [
    926, 152], "y": [1009, 112]}  # format: "band":[mean wavelength, bandwidth]
bandcolours = {"g": "tab:green", "r": "tab:orange",
               "i": "tab:red", "z": "tab:purple"}  # some colours for plotting



def plot_width_vs_z(fitband='z', average=False, secondband='i', prepeak=False, nonderedshifted=False):
    ''' Plots the width vs redshift for the z-band fit data. 
    Parameters
    ----------
    fitband : str
    average : bool
        If True, we'll plot the average of two bands of width data
    secondband : str
        If average == True, this is the second band that we'll use in the averaging. 
    prepeak : bool
        If true, will plot the fits that require pre-peak lightcurve data. 
    '''
    if nonderedshifted:
        suffix = "_non_deredshifted"
    elif prepeak:
        suffix = "_PREPEAK"
    else:
        suffix = ""
    file_location = "Images/Pre-peak Images/" if prepeak else "Images/"
    
    # load in data from our fit band
    prefix = filenames[fitband] + "_"
    with open(prefix + 'pickled_data' + suffix, 'rb') as file:
        data = pickle.load(file)

    if average:
        # now load in data from our second fitting band
        prefix = filenames[secondband] + "_"
        with open(prefix + 'pickled_data' + suffix, 'rb') as file:
            data2 = pickle.load(file)
        
        # now get all of the like data between the two data objects
        _, zs1, widths1, werr1, _, _, widths2, werr2 = Methods.get_like_widths(data, data2)
        
        zs = zs1    # need the redshifts for the plotting!
        widths = np.mean([widths1, widths2], axis=0)    # get the mean in the widths across the two bands
        werr = np.array([np.sqrt(werr1[i]**2 + werr2[i]**2) / 2 for i in range(len(werr1))])    # calculate the error in the two values from gaussian error propagation
    else: 
        zs = data["z"].to_numpy()
        widths = data["Width"].to_numpy()
        werr = data["Width_err"].to_numpy()
        
        # only care about values that have error less than the signal
        err_inds = np.less(werr, widths)
        zs = zs[err_inds]
        widths = widths[err_inds]
        werr = werr[err_inds]
        
    # find the best fit (chi square minimisation by default) of the width vs 1+z curve
    best_fit, cov = opt.curve_fit(power_1z, 1 + zs, widths, p0=(1,),
                                  sigma=werr, absolute_sigma=True)
    best_fit = best_fit[0]  # this is our b parameter
    err = np.sqrt(np.diag(cov))[0]  # covariance matrix error
    print(f"b = {best_fit} \pm {err}")
    red_chisquare = sum([(((1 + zs[i])**best_fit - widths[i]) / werr[i])**2 for i in range(len(widths))]) / (len(widths) - 1)
    print(f"fitband={fitband}, prepeak={prepeak}, nonderedshifted={nonderedshifted}, reduced chi square = {red_chisquare:.3f}, averaged={average}, secondband={secondband}")

    # now calculate the binned data for visualisation purposes
    binX, binY, binYerr = Methods.bin_data(1 + zs, widths, yerr=werr, n=50)

    fig, ax = plt.subplots(figsize=(9, 5))
    # plot the raw width data
    ax.errorbar(1 + zs, widths, yerr=werr,
                fmt='.', c='tab:gray', alpha=0.3, label='Data', rasterized=True)
    # now plot the errorbars on top
    ax.errorbar(binX, binY, yerr=binYerr, fmt='.', c='tab:red',
                label='Binned Data', rasterized=True)
    # plot a 1 + z line
    line = np.linspace(1, 1.01 + max(zs), 5)
    ax.plot(line, line, c='k', ls=':', label='$1 + z$')
    # now plot the line of best fit
    rbest = str(round(best_fit, 3))
    rerr = str(round(err, 3))
    ax.plot(line, power_1z(line, best_fit), c='tab:blue',
            ls='--', label='$(1 + z)^{'+rbest+'\pm'+rerr+'}$')
    # now plot the uncertainty in the line of best fit
    for sign in [1, -1]:
        ax.plot(line, power_1z(line, best_fit + sign * err),
                c='tab:blue', ls=':', alpha=0.7)
    ax.set_ylim(0.4, 4)
    ax.set_xlim(1, 1.01 + max(zs))
    ax.set_xlabel(r"$1 + z$")
    ax.set_ylabel("Lightcurve width $w$")
    ax.legend()
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    if average:
        fig.savefig(f"{file_location}{fitband}-{secondband}AveWidth-vs-1+z-first{len(zs)}{suffix}.png",
                    dpi=400, bbox_inches='tight')
        fig.savefig(f"{file_location}{fitband}-{secondband}AveWidth-vs-1+z-first{len(zs)}{suffix}.pdf",
                    dpi=400, bbox_inches='tight')
    else:
        fig.savefig(f"{file_location}{prefix}Width-vs-1+z-first{len(zs)}{suffix}.png", dpi=400, bbox_inches='tight')
        fig.savefig(f"{file_location}{prefix}Width-vs-1+z-first{len(zs)}{suffix}.pdf", dpi=400, bbox_inches='tight')

def plot_averaged_widths(prepeak=False):
    ''' Plots the width vs redshift for the z-band fit data. 
    Parameters
    ----------
    fitband : str
    average : bool
        If True, we'll plot the average of two bands of width data
    secondband : str
        If average == True, this is the second band that we'll use in the averaging. 
    prepeak : bool
        If true, will plot the fits that require pre-peak lightcurve data. 
    '''
    suffix = "_PREPEAK" if prepeak else ""
    file_location = "Images/Pre-peak Images/" if prepeak else "Images/"
    
    # load in data from our fit band
    with open(filenames['g'] + '_pickled_data' + suffix, 'rb') as file:
        data1 = pickle.load(file)
    with open(filenames['r'] + '_pickled_data' + suffix, 'rb') as file:
        data2 = pickle.load(file)
    with open(filenames['i'] + '_pickled_data' + suffix, 'rb') as file:
        data3 = pickle.load(file)
    with open(filenames['z'] + '_pickled_data' + suffix, 'rb') as file:
        data4 = pickle.load(file)
        
    # cids1, cids2, cids3, cids4 = data1["CID"].to_numpy(), data2["CID"].to_numpy(), data3["CID"].to_numpy(), data4["CID"].to_numpy()
    allSNe = {}
    data_list = [data1, data2, data3, data4]
    for i, data in enumerate(data_list):
        for j, cid in enumerate(data["CID"].to_numpy()):
            if cid not in allSNe:
                if data["Width_err"][j] < data["Width"][j]:
                    allSNe[cid] = [1, data["z"][j], data["Width"][j], data["Width_err"][j]**2]
            else:
                if data["Width_err"][j] < data["Width"][j]:
                    allSNe[cid] = [allSNe[cid][0] + 1, data["z"][j], allSNe[cid][2] + data["Width"][j], allSNe[cid][3] + data["Width_err"][j]**2]
    
    widths = np.array([allSNe[cid][2] / allSNe[cid][0] for cid in allSNe])
    zs = np.array([allSNe[cid][1] for cid in allSNe])
    werr = np.array([np.sqrt(allSNe[cid][3]) / allSNe[cid][0] for cid in allSNe])
    colours = np.array([allSNe[cid][0] for cid in allSNe])
    
    
    # args = np.argwhere((zs < 0.8) & (zs > 0.21)).flatten()
    # widths = widths[args]
    # zs = zs[args]
    # werr = werr[args]
    # colours = colours[args]
    
    
    import matplotlib as mpl
    from matplotlib.colors import LinearSegmentedColormap, ListedColormap
    cmap = ListedColormap(["#f98e08", "#a1a418", "#53a45b", "#1e998a"]) 
    alpha = 0.45
    colours2 = []
    maxval = max(colours)
    for i in range(len(colours)):
        # print(colours[i] / maxval)
        colours2.append(cmap((colours[i] - 1) / maxval))
        
    # find the best fit (chi square minimisation by default) of the width vs 1+z curve
    best_fit, cov = opt.curve_fit(power_1z, 1 + zs, widths, p0=(1,),
                                  sigma=werr, absolute_sigma=True)
    best_fit = best_fit[0]  # this is our b parameter
    err = np.sqrt(np.diag(cov))[0]  # covariance matrix error
    print(f"b = {best_fit} \pm {err}")
    red_chisquare = sum([(((1 + zs[i])**best_fit - widths[i]) / werr[i])**2 for i in range(len(widths))]) / (len(widths) - 1)
    print(f"all band average, prepeak={prepeak}, reduced chi square = {red_chisquare:.4f}")
    
    # find the best fit (chi square minimisation by default) of the width vs 1+z curve (assuming a linear case)
    lin_best_fit, lin_cov = opt.curve_fit(linear_model, 1 + zs, widths, p0=(1, 0),
                                  sigma=werr, absolute_sigma=True)
    lin_m, lin_c = lin_best_fit
    lin_m_err, lin_c_err = np.sqrt(np.diag(lin_cov))  # covariance matrix error
    print(f"w = ({lin_m:.4f} \pm {lin_m_err:.4f}) * (1 + z) + ({lin_c:.4f} \pm {lin_c_err:.4f})")
    lin_red_chisquare = sum([((lin_m * (1 + zs[i]) + lin_c - widths[i]) / werr[i])**2 for i in range(len(widths))]) / (len(widths) - 1)
    print(f"all band average, prepeak={prepeak}, LINEAR reduced chi square = {lin_red_chisquare:.4f}")

    
    # now calculate the binned data for visualisation purposes
    binX, binY, binYerr = Methods.bin_data(1 + zs, widths, yerr=werr, n=50)

    fig, ax = plt.subplots(figsize=(9, 5))
    # plot the raw width data
    # width_plot = ax.errorbar(1 + zs, widths, yerr=werr,
    #             fmt='.', color=colours, alpha=0.3, label='Data', rasterized=True)
    width_plot = ax.errorbar(1 + zs, widths, yerr=werr,
                fmt=',', ecolor=colours2, alpha=alpha/2, label='Data', rasterized=True)
    scatter_plot = ax.scatter(1 + zs, widths, c=colours, cmap=cmap, vmin=1, vmax=4, s=2, alpha=alpha, rasterized=True)
    # legend1 = ax.legend(*width_plot.legend_elements(), title="Averaged Bands")
    # now plot the errorbars on top
    ax.errorbar(binX, binY, yerr=binYerr, fmt='.', c='tab:red',
                label='Binned Data', rasterized=True)
    # plot a 1 + z line
    line = np.linspace(1, 1.01 + max(zs), 5)
    ax.plot(line, line, c='k', ls=':', label='$1 + z$')
    # now plot the line of best fit
    rbest = f"{best_fit:.3f}"
    rerr = f"{err:.3f}"
    ax.plot(line, power_1z(line, best_fit), c='tab:blue',
            ls='--', label='$(1 + z)^{'+rbest+'\pm'+rerr+'}$')
    # now plot the uncertainty in the line of best fit
    for sign in [1, -1]:
        ax.plot(line, power_1z(line, best_fit + sign * err),
                c='tab:blue', ls=':', alpha=0.7)
    ax.set_ylim(0.5, 3)
    ax.set_xlim(1, 1.01 + max(zs))
    ax.set_xlabel(r"$1 + z$")
    ax.set_ylabel("Lightcurve width $w$")
    ax.legend(loc='upper left')
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    colourbar = fig.colorbar(scatter_plot, label='Number of Averaged Bands', pad=0, aspect=40, ticks=[4/3, 12.5/6, 17.4/6, 10.9/3])
    colourbar.solids.set(alpha=1)
    colourbar.ax.set_yticklabels(['1', '2', '3', '4'])
    fig.savefig(f"{file_location}AllAveWidths-vs-1+z-first{len(zs)}{suffix}.png", dpi=400, bbox_inches='tight')
    fig.savefig(f"{file_location}AllAveWidths-vs-1+z-first{len(zs)}{suffix}.pdf", dpi=400, bbox_inches='tight')
    
    # now print out the total dispersion about the best fit
    errs = widths - (1 + zs)**best_fit
    print("Dispersion in fit widths compared to best (1 + z)^b trend is:", np.std(errs))

def plot_all_widths(prepeak=False, nonderedshifted=False):
    ''' Plots the width vs 1 + z data for all 4 bands on a 2x2 plot. 
    Parameters
    ----------
    prepeak : bool
        If true, will plot the fits that require pre-peak lightcurve data. 
    '''
    if nonderedshifted:
        suffix = "_non_deredshifted"
    elif prepeak:
        suffix = "_PREPEAK"
    else:
        suffix = ""
    file_location = "Images/Pre-peak Images/" if prepeak else "Images/"
    ### for more in depth documentation, check out the plot_width_vs_z function above
    fig, axes = plt.subplots(nrows=2, ncols=2, sharex='col', sharey='row', gridspec_kw={'hspace':0, 'wspace':0},
                             figsize=(10, 8))
    axes = axes.flatten()
    # 0 - top left, 1 - top right, 2 - bottom left, 3 - bottom right
    
    for i, band in enumerate(['g', 'r', 'i', 'z']):
        prefix = filenames[band] + "_"
        with open(prefix + 'pickled_data' + suffix, 'rb') as file:
            data = pickle.load(file)
            
        # unpack the data
        zs = data["z"].to_numpy()
        widths = data["Width"].to_numpy()
        werr = data["Width_err"].to_numpy()
        
        # only care about values that have error less than the signal
        err_inds = np.less(werr, widths)
        zs = zs[err_inds]
        widths = widths[err_inds]
        werr = werr[err_inds]
        
        # calculate line of best fit
        best_fit, cov = opt.curve_fit(power_1z, 1 + zs, widths, p0=(1,),
                                      sigma=werr, absolute_sigma=True)
        best_fit = best_fit[0]
        err = np.sqrt(np.diag(cov))[0]
        print(f"b = {best_fit} \pm {err}")
        
        # calculate binned data for visualisation
        binX, binY, binYerr = Methods.bin_data(1 + zs, widths, yerr=werr, n=50)
        
        # only need to create the complete axes (with labels) once, so have a check for this.
        if i == 0:
            errlab = 'Data'; binlab = 'Binned Data'; expectlab = '$1+z$'; notimedillab = "No Time Dil"
        else:
            errlab = ''; binlab = ''; expectlab = ''; notimedillab = ""
        axes[i].errorbar(1 + zs, widths, yerr=werr,
                    fmt='.', c='tab:gray', alpha=0.3, label=errlab, rasterized=True)
        axes[i].errorbar(binX, binY, yerr=binYerr, fmt='.', c='tab:red',
                    label=binlab, rasterized=True)
        line = np.linspace(1, 2.15, 5)
        axes[i].plot(line, line, c='k', ls=':', label=expectlab)
        rbest = f"{best_fit:.3f}"
        rerr = f"{err:.3f}"
        axes[i].plot(line, power_1z(line, best_fit), c='tab:blue',
                ls='--', label='$(1 + z)^{'+rbest+'\pm'+rerr+'}$')
        for sign in [1, -1]:
            axes[i].plot(line, power_1z(line, best_fit + sign * err),
                    c='tab:blue', ls=':', alpha=0.7)
            
        if nonderedshifted:
            axes[i].plot(line, np.ones(len(line)), ls='--', c='k', label=notimedillab)
            axes[i].set(xlim=(1, 2.15), ylim=(0.4, 2.1))
            axes[i].text(2, 1.8, f'${band}$', fontsize=18)
        else:
            axes[i].set(xlim=(1, 2.15), ylim=(0.4, 3.5))
            axes[i].text(2, 3.15, f'${band}$', fontsize=18)
        
        if i in [0, 1]:
            axes[i].tick_params(axis='x', which='both', direction="in", top=True)
        if i in [2, 3]:
            axes[i].set_xlabel(r"$1 + z$")
            axes[i].tick_params(axis='x', which='major', length=6, direction="inout", top=True)
            axes[i].tick_params(axis='x', which='minor', length=4, direction="inout", top=True)
        if i in [0, 2]:
            axes[i].set_ylabel("Lightcurve width $w$")
            axes[i].tick_params(axis='y', which='major', length=6, direction="inout", right=True)
            axes[i].tick_params(axis='y', which='minor', length=4, direction="inout", right=True)
        if i in [1, 3]:
            axes[i].tick_params(axis='y', which='both', direction="in", right=True)
        axes[i].legend(loc='upper left')
        axes[i].yaxis.set_minor_locator(AutoMinorLocator())
        axes[i].xaxis.set_minor_locator(AutoMinorLocator())
    
    fig.savefig(f"{file_location}All-Widths-vs-1+z{suffix}.png", dpi=400, bbox_inches='tight')
    fig.savefig(f"{file_location}All-Widths-vs-1+z{suffix}.pdf", dpi=400, bbox_inches='tight')



def plot_dispersion_vs_scaling(band, method='power'):
    ''' Plots the reference curve flux dispersion as a function of the scaling parameter for a given scaling method. 
    Parameters
    ----------
    method : str
        One of 'power' or 'linear' that decides which dispersion-vs-scaling data we're plotting.
    '''
    # load in the data
    with open(method + f"_dispersion_pickle_{band}", 'rb') as file:
        disps = pickle.load(file)
    bN = len(disps[0, :])   # this is the number of scaling parameters we've used

    # get the median dispersion value for each b across all SN iterations
    med_disps, std_disps = np.zeros(bN), np.zeros(bN)
    for i in range(bN):
        arr = disps[:, i][np.where(disps[:, i] > 0)]
        med_disps[i] = np.median(arr)
        # std_disps[i] = np.std(arr)
    # med_disps = np.array([np.median(disps[:, j]) for j in range(bN)])
    # calculate the standard deviation according to gaussian uncertainty propagation laws
    std_disps = np.array([np.sqrt(np.sum(
        [sigma**2 for sigma in disps[:, i]])) / len(disps[:, 0]) for i in range(bN)])
    # std_disps = np.array([np.std(disps[:, i]) for i in range(bN)])
    # now create an array for our scaling variation parameter
    if method == 'linear':
        bs = np.linspace(0, 2, bN)
    else:   # power
        bs = np.linspace(-1, 2, bN)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.errorbar(bs, med_disps, yerr=std_disps, rasterized=True)
    ax.set_xlabel("$b$")
    ax.set_ylabel("Reference Flux Dispersion")
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    fig.savefig(f'Images/DispersionScaling-{method}_{band}.png', dpi=400, bbox_inches='tight')
    fig.savefig(f'Images/DispersionScaling-{method}_{band}.pdf', dpi=400, bbox_inches='tight')
    

def plot_all_dispersion_vs_scaling():
    ''' Plots the reference curve flux dispersion as a function of the scaling parameter across all of the bands. 
    '''
    all_disps, all_std_disps = np.zeros(30), np.zeros(30)
    # load in the data
    bands = ['r', 'i', 'z']
    for band in bands:
        with open(f"power_dispersion_pickle_{band}", 'rb') as file:
            disps = pickle.load(file)
        bN = len(disps[0, :])   # this is the number of scaling parameters we've used
    
        # get the median dispersion value for each b across all SN iterations
        med_disps, std_disps = np.zeros(bN), np.zeros(bN)
        for i in range(bN):
            arr = disps[:, i][np.where(disps[:, i] > 0)]
            med_disps[i] = np.median(arr)
            # std_disps[i] = np.std(arr)
        # med_disps = np.array([np.median(disps[:, j]) for j in range(bN)])
        # calculate the standard deviation according to gaussian uncertainty propagation laws
        std_disps = np.array([np.sqrt(np.sum(
            [sigma**2 for sigma in disps[:, i]])) / len(disps[:, 0]) for i in range(bN)])
        all_disps += med_disps
        all_std_disps += std_disps**2
    
    all_disps /= len(bands)
    all_std_disps = np.sqrt(all_std_disps) / len(bands)
    # now create an array for our scaling variation parameter
    bs = np.linspace(-1, 2, bN)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.errorbar(bs, all_disps, yerr=all_std_disps, rasterized=True)
    ax.set_xlabel("$b$")
    ax.set_ylabel("Reference Flux Dispersion")
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    fig.savefig('Images/DispersionScaling-power-'+''.join([band for band in bands])+'bands.png', dpi=400, bbox_inches='tight')
    fig.savefig('Images/DispersionScaling-power-'+''.join([band for band in bands])+'bands.pdf', dpi=400, bbox_inches='tight')



def compare_widths(band1, band2, prepeak=False):
    '''
    Parameters
    ----------
    band1 : str
    band2 : str
    prepeak : bool
        If true, will plot the fits that require pre-peak lightcurve data. 
    '''
    suffix = "_PREPEAK" if prepeak else ""
    file_location = "Images/Pre-peak Images/" if prepeak else "Images/"
    prefix1 = filenames[band1] + "_"
    prefix2 = filenames[band2] + "_"
    with open(prefix1 + 'pickled_data' + suffix, 'rb') as file:
        data1 = pickle.load(file)
    with open(prefix2 + 'pickled_data' + suffix, 'rb') as file:
        data2 = pickle.load(file)
        
    _, _, widths1, werr1, _, _, widths2, werr2 = Methods.get_like_widths(data1, data2)

    xdata, xerr = widths1, werr1
    ydata, yerr = widths2, werr2

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.errorbar(xdata, ydata, xerr=xerr, yerr=yerr,
                rasterized=True, c='tab:gray', alpha=0.1, fmt='.', label='Raw Widths')
    binX, binY, binXerr, binYerr = Methods.bin_data(
        xdata, ydata, xerr=xerr, yerr=yerr, n=50)
    ax.errorbar(binX, binY, xerr=binXerr, yerr=binYerr,
                c='tab:red', rasterized=True, fmt='.', label='Binned Widths')
    x = np.array([0, 1.1 * max([max(xdata), max(ydata)])])
    ax.set_xlim(0.7, min(3, 1.1 * max(xdata)))
    ax.set_ylim(0.7, min(3, 1.1 * max(ydata)))
    ax.plot(x, x, c='k', ls='--')
    ax.set(xlabel=f"${band1}$ Band Width", ylabel=f"${band2}$ Band Width")
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    
    ## Below is some linear regression code in case we want to fit the relationship between the bands
    # [m, c], cov = opt.curve_fit(linear_model, xdata, ydata, p0=(1, 0),
    #                           sigma=(yerr), absolute_sigma=True)
    # err = np.sqrt(np.diag(cov))
    # mbest = str(round(m, 3))
    # cbest = str(round(c, 3))
    # merr = str(round(err[0], 3))
    # cerr = str(round(err[1], 3))
    # ax.plot(x, linear_model(x, m, c), c='tab:blue', ls='--', label='$(' + mbest + '\pm' + merr + ')x + ' + cbest + '\pm' + cerr + '$')
    
    ax.legend()
    fig.savefig(f"{file_location}{band1}-{band2} Width Comparison{suffix}.png",
                dpi=400, bbox_inches='tight')
    fig.savefig(f"{file_location}{band1}-{band2} Width Comparison{suffix}.pdf",
                dpi=400, bbox_inches='tight')



def compare_all_widths(prepeak=False):
    '''
    Parameters
    ----------
    prepeak : bool
        If true, will plot the fits that require pre-peak lightcurve data. 
    '''
    suffix = "_PREPEAK" if prepeak else ""
    file_location = "Images/Pre-peak Images/" if prepeak else "Images/"
    fig, axes = plt.subplots(nrows=3, sharex=True, gridspec_kw={'hspace':0}, figsize=(3, 9))
    band1 = 'i'
    
    for i, band2 in enumerate(['g', 'r', 'z']):
        prefix1 = filenames[band1] + "_"
        prefix2 = filenames[band2] + "_"
        with open(prefix1 + 'pickled_data' + suffix, 'rb') as file:
            data1 = pickle.load(file)
        with open(prefix2 + 'pickled_data' + suffix, 'rb') as file:
            data2 = pickle.load(file)
            
        _, _, widths1, werr1, _, _, widths2, werr2 = Methods.get_like_widths(data1, data2)
    
        xdata, xerr = widths1, werr1
        ydata, yerr = widths2, werr2
    
        axes[i].errorbar(xdata, ydata, xerr=xerr, yerr=yerr, rasterized=True, c='tab:gray', alpha=0.1, 
                         fmt='.', label='Raw Widths')
        binX, binY, binXerr, binYerr = Methods.bin_data(xdata, ydata, xerr=xerr, yerr=yerr, n=50)
        axes[i].errorbar(binX, binY, xerr=binXerr, yerr=binYerr, c='tab:red', rasterized=True, 
                         fmt='.', label='Binned Widths')
        x = np.array([0, 4])
        axes[i].plot(x, x, c='k', ls='--')
        axes[i].set(ylabel=f"${band2}$ Band Width", xlim=(0.7, 3), 
                    ylim=(0.7, 3))
            
        axes[i].yaxis.set_minor_locator(AutoMinorLocator())
        axes[i].xaxis.set_minor_locator(AutoMinorLocator())
        
        if i == 0:
            axes[i].legend(loc='upper right')
        if i in [0, 1]:
            axes[i].tick_params(axis='x', which='both', direction="in", top=True)
        if i == 2:
            axes[i].set_xlabel("$i$ Band Width")
            axes[i].tick_params(axis='x', which='major', length=6, direction="inout", top=True)
            axes[i].tick_params(axis='x', which='minor', length=4, direction="inout", top=True)

    fig.savefig(f"{file_location}All_Width_Comparison{suffix}.png", dpi=400, bbox_inches='tight')
    fig.savefig(f"{file_location}All_Width_Comparison{suffix}.pdf", dpi=400, bbox_inches='tight')



def plot_lightcurve_bands(cid, z, save=True):
    ''' Plots the data of a lightcurve in all (available) bands, with its SALT fit overlaid. 
    Parameters
    ----------
    cid : int
        The SNe CID
    z : int
        The redshift of the supernova (for the filename)
    save : bool
        If True, saves a .png and .pdf of the figure
    '''
    fig, ax = plt.subplots(figsize=(5, 4))
    for band in ['g', 'r', 'i', 'z']:
        tdata, data, data_err = Methods.get_SN_data(cid, band)  # get the data
        tsim, sim, sim_err = Methods.get_SN_sim(cid, band)  # get the SALT fit

        if len(tdata) == 0:  # if there is no data in this band, move on with the loop
            continue

        data /= max(sim)
        data_err /= max(sim)
        sim /= max(sim)  # normalize the data

        ax.errorbar(tdata, data, yerr=data_err, fmt='.',
                    color=bandcolours[band], label=f"${band}$-band data")
        ax.plot(tsim, sim, c=bandcolours[band])

    ax.set(xlabel=r"Time Since Peak Brightness, $t - t_{\mathrm{peak}}$ (days)",
           ylabel="Normalised Flux", xlim=(1.4 * min(tdata), 1.2 * max(tdata)))
    ax.legend()
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(which='major')

    if save:
        fig.savefig(f"Images/{cid}-z{z}-Lightcurve.png", dpi=400, bbox_inches='tight')
        fig.savefig(f"Images/{cid}-z{z}Lightcurve.pdf", dpi=400, bbox_inches='tight')



def plot_stretch(cid, z, fit_band, width_frac=1/16):
    '''  Plots the reference curve sampling and width fitting for a given SNe
    Parameters
    ----------
    cid : int
        The SNe CID
    z : int
        The redshift of the supernova
    fit_band : str
        The filter of the *individual* light curves that we want to fit
    width_frac : float
        The width in the reference curve sampling that we'll accept.
    '''
    fit_band_center, fit_band_width = DES_filters[fit_band]     # get the fitting band params
    # the wiggle room either side of the fitting-band mean wavelength
    wiggle = width_frac * fit_band_width

    # make a 2x1 figure
    fig, axes = plt.subplots(ncols=2, gridspec_kw={'width_ratios': [1, 2]},
                             figsize=(10, 5))
    plot_z = np.array([0, max(Zs)])
    # the left plot axes[0] will have the sampling regions in terms of source/reference redshift
    # start by plotting each of these regions
    for band in ['g', 'r', 'i', 'z']:
        band_center = DES_filters[band][0]
        
        upper, lower = (band_center * (1 + plot_z) + wiggle) / fit_band_center - 1, (band_center * (1 + plot_z) - wiggle) / fit_band_center - 1
        axes[0].plot(plot_z, upper, c=bandcolours[band], label=f"${band}$ Band")
        axes[0].plot(plot_z, lower, c=bandcolours[band])
        axes[0].fill_between(plot_z, upper, lower, color=bandcolours[band], alpha=0.2)
        
    axes[0].set_ylim(0, 1.45)
    axes[0].set_xlim(0, max(Zs))
    axes[0].yaxis.set_minor_locator(AutoMinorLocator())
    axes[0].xaxis.set_minor_locator(AutoMinorLocator())
    axes[0].scatter([z] * len(Zs), Zs, c='k', s=0.1, rasterized=True)
    axes[0].set(xlabel="Target SNe $z$", ylabel="Reference $z$")
    axes[0].legend(loc='upper left')
    axes[0].grid(which='major')

    # the right plot will have our constructed reference curve. start by getting the reference
    used_lab = {'g': 0, 'r': 0, 'i': 0, 'z': 0}; used_bands = []
    all_tdata, all_data, bands, _, valid_cids, _, _, running_index = Methods.get_reference_curve(
        cid, z, fit_band, width_frac=width_frac, full_return=True)

    # now plot the reference curve in terms of each data points colour
    run_total = 0   # this is a running total of the index (data points) that we're at
    for j, vcid in enumerate(valid_cids):
        i1, i2 = run_total, run_total + running_index[j]    # our left/right bounds for indexing
        if used_lab[bands[j]] == 0:     # if we haven't used a label yet...
            p = axes[1].scatter(all_tdata[i1:i2], all_data[i1:i2],
                                c=bandcolours[bands[j]], s=2, rasterized=True)
            used_lab[bands[j]] = p
            used_bands += [bands[j]]
        else:   # if we've already used a label.
            axes[1].scatter(all_tdata[i1:i2], all_data[i1:i2],
                            c=bandcolours[bands[j]], s=2, rasterized=True)
        run_total += running_index[j]   # update running total

    axes[1].set_ylim(-0.2, 1.5)
    axes[1].set(xlabel="Time Since Peak Brightness (days)",
                ylabel="Normalised Flux")
    
    # now to plot the source data on top of the reference curve
    tsim, sim, _ = Methods.get_SN_sim(cid, fit_band)     # get SALT fit
    tdata, data, data_err = Methods.get_SN_data(cid, fit_band)   # get data
    data /= max(sim)
    data_err /= max(sim)  # normalise the data based on the SALT fit
    tdata = tdata.to_numpy()
    data = data.to_numpy()
    data_err = data_err.to_numpy()  # convert those pandas dataframes into arrays

    use_indices = np.where(tdata <= max(all_tdata))
    tdata = tdata[use_indices]
    data = data[use_indices]
    data_err = data_err[use_indices]

    p2 = axes[1].errorbar(tdata, data, yerr=data_err, c='tab:blue', fmt='.', markersize=10, label="Raw Data",
                          rasterized=True)

    width, _, _, _ = StretchMethod.get_stretch(cid, z, fit_band)
    print(width)

    p3 = axes[1].errorbar(tdata / width, data, yerr=data_err, c='k', fmt='.', markersize=10, label="Scaled Data",
                          rasterized=True)

    axes[1].legend()
    # getting the reference curve legend entries to look nice is a little confusing.
    axes[1].legend([tuple(used_lab[used_band] for used_band in used_bands), p2, p3], ["Reference Photometry", "Target Photometry (Obs. Frame)", "Target Photometry (Scaled)"],
                   handler_map={tuple: HandlerTuple(ndivide=None)})

    axes[1].yaxis.set_minor_locator(AutoMinorLocator())
    axes[1].xaxis.set_minor_locator(AutoMinorLocator())
    
    fig.savefig(f"Images/StretchPlot-{cid}-z{z}.png", dpi=400, bbox_inches='tight')
    fig.savefig(f"Images/StretchPlot-{cid}-z{z}.pdf", dpi=400, bbox_inches='tight')
    # fig.savefig(f"Images/StretchPlot-{cid}-z{z}-{fit_band}band.png", dpi=400, bbox_inches='tight')
    # fig.savefig(f"Images/StretchPlot-{cid}-z{z}-{fit_band}band.pdf", dpi=400, bbox_inches='tight')



def plot_reference_creation(cid, z, fit_band):
    ''' Plots the reference curve before and after time dilation correction.
    Parameters
    ----------
    cid : int
        The SNe CID
    z : int
        The redshift of the supernova
    fit_band : str
        One of 'g', 'r', 'i', 'z' that says what the source SNe band is. 
    '''
    # make a 1x2 figure with shared x axis
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(5, 8), sharex=True, gridspec_kw={"hspace": 0})

    # want to plot the same thing twice, the first one non-scaled and the second scaled
    for i, scale in enumerate([False, True]):
        used_lab = {'g': 0, 'r': 0, 'i': 0, 'z': 0}     # a dict to show if we've used the label in each band yet
        all_tdata, all_data, bands, _, valid_cids, valid_zs, _, running_index = Methods.get_reference_curve(
            cid, z, fit_band, scale=scale, full_return=True)    # get the full reference curve data

        run_total = 0   # running total to say how many points we've looked at so far
        for j, vcid in enumerate(valid_cids):
            i1, i2 = run_total, run_total + running_index[j]    # get the lower and upper data indices for plotting
            if used_lab[bands[j]] == 0:     # if we havent used the label yet, lets use it!
                lab_z = DES_filters[bands[j]][0] * (1 + z) / DES_filters[fit_band][0] - 1
                lab = f"$z\sim {lab_z:.2f}$ (${bands[j]}$ band)"
                used_lab[bands[j]] = 1
            else:
                lab = ''
            axes[i].scatter(all_tdata[i1:i2], all_data[i1:i2], c=bandcolours[bands[j]], s=2, rasterized=True,
                            label=lab)  # plot the data with the colour of its respective band
            run_total += running_index[j]   # update running total

    for ax in axes:
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.set_ylim(-0.2, 1.5)
        ax.grid()
    axes[0].tick_params(axis='x', which='both', direction="in", top=True)
    axes[1].tick_params(axis='x', which='major', length=6, direction="inout", top=True)
    axes[1].tick_params(axis='x', which='minor', length=4, direction="inout", top=True)
    axes[0].set_ylabel("Normalised Flux (Pre-Correction)")
    axes[1].set(xlabel=r"Time Since Peak Brightness, $t - t_{\mathrm{peak}}$ (days)",
                ylabel="Normalised Flux (Corrected)")

    axes[1].legend(loc="upper right")
    
    fig.savefig("Images/Reference_Construction.png", dpi=400, bbox_inches='tight')
    fig.savefig("Images/Reference_Construction.pdf", dpi=400, bbox_inches='tight')



def plot_reference_pops(fit_band):
    ''' Plots the reference populations across scaling parameters and redshifts for a given fitting band. 
    Parameters
    ----------
    fit_band : str
        One of 'g', 'r', 'i', 'z'
    '''
    import matplotlib.colors as colors
    # load in the data
    with open(f'reference_pops_{fit_band}band', 'rb') as file:
        cids, zs, width, reference_pops = pickle.load(file)

    reference_pops[reference_pops == 0] = 1     # since we're doing it on a log scale, need to convert all 0s to 1s

    fig, ax = plt.subplots()

    p = np.argsort(zs)  # will need to sort the zs so that the figure is consistent up/down
    reference_pops = reference_pops[p]
    zs = zs[p]
    cids = cids[p]
    
    # now plot the colourmesh on the axes with log width, log population, and linear interpolation shading
    pcm = ax.pcolormesh(np.log2(width), zs, reference_pops,
                        norm=colors.LogNorm(vmin=reference_pops.min(), vmax=reference_pops.max()),
                        rasterized=True)
    fig.colorbar(pcm, ax=ax, extend='both', label="Reference Population")

    ax.invert_xaxis()   # want to go from wide wiggle to narrow wiggle
    # i was having trouble above with an automatic log2 scale, so I need to manually define the ticks
    ax.set_xticks(np.log2(width))
    ax.set_xticklabels([f"$2^{{{int(np.log2(wid))}}}$" for wid in width])

    ax.yaxis.set_minor_locator(AutoMinorLocator())

    ax.set(xlabel=r"Width Factor, $\delta$", ylabel="Target SNe Redshift $z$")

    fig.savefig(f"Images/ReferencePopulations_{fit_band}band.png", dpi=400, bbox_inches='tight')
    fig.savefig(f"Images/ReferencePopulations_{fit_band}band.pdf", dpi=400, bbox_inches='tight')

def plot_flux_vs_error():
    ''' Plots all of the DES photometry on a flux vs error plot, with points coloured by redshift. 
    '''
    fig, ax = plt.subplots()
    # get all of the flux and error points for SNe that are in our final sample
    flux_err = Methods.lc_uncleaned["FLUXCAL_ERR"][np.isin(Methods.lc_uncleaned["CID"], Methods.CIDs)].to_numpy()
    flux = Methods.lc_uncleaned["FLUXCAL"][np.isin(Methods.lc_uncleaned["CID"], Methods.CIDs)].to_numpy()
    CIDZs = {cid: Methods.Zs[i] for i, cid in enumerate(Methods.CIDs)}  # assign each CID to a redshift
    zs = np.array([CIDZs[cid] for cid in Methods.lc_uncleaned["CID"] if cid in Methods.CIDs])   # now get the redshift for each datapoint
    sort_inds = np.argsort(zs).flatten()    # to make sure we can still see low brightness, high z points, plot the low z data first
    zs = zs[sort_inds]; flux = flux[sort_inds]; flux_err = flux_err[sort_inds]
    plot = ax.scatter(flux_err, flux, c=zs, s=0.5, cmap='viridis', alpha=0.25, rasterized=True)
    ax.set(xlim=[0, 50], ylim=[-100, 500], xlabel='Flux Uncertainty', ylabel='Flux')
    colourbar = fig.colorbar(plot, label='Redshift $z$')
    colourbar.solids.set(alpha=1)   # make the colourbar scale opaque 
    ax.axvline(20, linestyle='--', c='tab:red')     # showing our error cut
    
    fig.savefig('Images/Flux_vs_Err.png', dpi=400, bbox_inches='tight')
    fig.savefig('Images/Flux_vs_Err.pdf', dpi=400, bbox_inches='tight')
    
def plot_2001V_spec():
    ''' Plots and saves the spectrum of 2001V from Matheson et al 2008 https://ui.adsabs.harvard.edu/abs/2008AJ....135.1598M/abstract
    '''
    DECam_data = np.genfromtxt("Images/DECamData.txt")
    data = np.genfromtxt("Images/sn2001v-20010225.txt")
    wavelengths, intensity = data[:, 0] / 10, data[:, 1]
    intensity /= max(intensity)
    every = 10
    alpha = 0.2
    for filterresponse in [True, False]:
        fig, ax = plt.subplots(figsize=(5, 4))
        if filterresponse == True:
            DESwavelengths = DECam_data[:, 0] / 10
            DESg = DECam_data[:, 1]
            DESr = DECam_data[:, 2]
            DESi = DECam_data[:, 3]
            DESz = DECam_data[:, 4]
            ax.plot(DESwavelengths, DESg, c='tab:green', label='DES $g$', rasterized=True)
            ax.fill_between(DESwavelengths, DESg, np.zeros(len(DESwavelengths)), color='tab:green', alpha=alpha, rasterized=True)
            ax.plot(DESwavelengths, DESr, c='tab:orange', label='DES $r$', rasterized=True)
            ax.fill_between(DESwavelengths, DESr, np.zeros(len(DESwavelengths)), color='tab:orange', alpha=alpha, rasterized=True)
            ax.plot(DESwavelengths, DESi, c='tab:red', label='DES $i$', rasterized=True)
            ax.fill_between(DESwavelengths, DESi, np.zeros(len(DESwavelengths)), color='tab:red', alpha=alpha, rasterized=True)
            # ax.plot(DESwavelengths, DESz, c='tab:purple')
            # ax.fill_between(DESwavelengths, DESz, np.zeros(len(DESwavelengths)), color='tab:purple', alpha=alpha)
        else:
            for band in ["g", "r", "i"]:
                positions = [DES_filters[band][0] + i * DES_filters[band][1] / 2 for i in [1, -1]]
                for p in positions:
                    ax.axvline(p, c=bandcolours[band])
                ax.fill_between(positions, np.ones(2), np.zeros(2), color=bandcolours[band], alpha=alpha, label=f'DES {band}', rasterized=True)
        
        ax.plot(wavelengths[::every], intensity[::every], lw=1, c='k', label='$z = z_1$', rasterized=True)
        ax.plot(wavelengths[::every] + 170, intensity[::every], lw=1, c='k', ls='--', label='$z = z_2$', rasterized=True)
        ax.set(xlabel='Wavelength (nm)', ylabel='Relative Transmittance/Intensity', xlim=[380, 870], ylim=[0, 1])
        ax.legend(loc='upper right')
        filename = "Images/SpectrumDESTransmittance" if filterresponse else "Images/SpectrumSimple"
        fig.savefig(filename + ".png", dpi=400, bbox_inches='tight')
        fig.savefig(filename + ".pdf", dpi=400, bbox_inches='tight')
        
def plot_err_floor_vs_MC(prepeak=False):
    
    suffix = "_PREPEAK" if prepeak else ""
    file_location = "Images/Pre-peak Images/" if prepeak else "Images/"
    
    # load in data from our fit band
    with open(filenames['g'] + '_pickled_data' + suffix, 'rb') as file:
        data1 = pickle.load(file)
    with open(filenames['r'] + '_pickled_data' + suffix, 'rb') as file:
        data2 = pickle.load(file)
    with open(filenames['i'] + '_pickled_data' + suffix, 'rb') as file:
        data3 = pickle.load(file)
    with open(filenames['z'] + '_pickled_data' + suffix, 'rb') as file:
        data4 = pickle.load(file)
    
    fig, ax = plt.subplots()
    
    data_list = [data1, data2, data3, data4]
    bands = ['g', 'r', 'i', 'z']
    
    for i, data in enumerate(data_list):
        sort = np.argsort(data['z'])
        ax.plot(1 + data['z'][sort], data['Width_err_floor'][sort], ls='--', c=bandcolours[bands[i]], label=bands[i], rasterized=True)
        ax.scatter(1 + data['z'][sort], data['Width_err_data'][sort], c=bandcolours[bands[i]], s=0.8, alpha=0.7, rasterized=True)
        
    ax.set(xlabel='$1 + z$', ylabel='Width Error', yscale='log')
    ax.legend()
    handles, labels = ax.get_legend_handles_labels()
    from matplotlib.lines import Line2D
    p1 = Line2D([0], [0], color='k', ls='--', label='Err Floor'); handles.append(p1)
    p2 = Line2D([0], [0], color='k', ls='None', marker='.', label='MC Err.'); handles.append(p2)
    ax.legend(handles=handles, loc='lower right')
    ax.grid()
    fig.savefig(file_location + 'Error-vs-z.png', dpi=400, bbox_inches='tight')
    fig.savefig(file_location + 'Error-vs-z.pdf', dpi=400, bbox_inches='tight')
    
def plot_w_vs_m(fitband='i'):
    file_location = "Images/"
    
    # load in data from our fit band
    prefix = filenames[fitband] + "_"
    with open(prefix + 'pickled_data', 'rb') as file:
        data = pickle.load(file)
    
    cids = data['CID']
    widths = data['Width']
    zs = data['z']
    
    _, intersect_inds, _ = np.intersect1d(CIDs, cids, return_indices=True)
    app_mag = mB[intersect_inds]
    
    fig, ax = plt.subplots()
    ax.scatter(app_mag, widths / (1 + zs), s=0.5, rasterized=True)
    ax.set(xlabel='Apparent Magnitude', ylabel='Recovered Stretch (w / (1 + z))', ylim=(0.5, 2))
    fig.savefig(file_location + "width_vs_apparentmag.png", dpi=400, bbox_inches='tight')
    fig.savefig(file_location + "width_vs_apparentmag.pdf", dpi=400, bbox_inches='tight')
    
def plot_binpop_vs_binwidth(fitband='i', N=10):
    file_location = "Images/"
    
    # load in data from our fit band
    prefix = filenames[fitband] + "_"
    with open(prefix + 'pickled_data', 'rb') as file:
        data = pickle.load(file)
    
    zs = data['z'].to_numpy()
    cids = data['CID'].to_numpy()
    
    samples = np.linspace(zs.min(), zs.max(), N)
    
    idx = np.array([np.argmin(np.abs(zs - sample)) for sample in samples])
    
    zs = zs[idx]
    cids = cids[idx]
    
    binwidths = [1, 2, 4, 6, 8]
    
    binpops = np.zeros((N, len(binwidths)))
    
    for i, (cid, z) in enumerate(zip(cids, zs)):
        all_tdata, all_data, bands = Methods.get_reference_curve(cid, z, fitband)
        
        for j, binwidth in enumerate(binwidths):
            steps = int((max(all_tdata - min(all_tdata))) / binwidth)
            x = np.linspace(min(all_tdata), max(all_tdata), steps) / (1 + z)
            arg_lens = np.zeros(len(x))
            for k in range(len(x)):
                args = np.argwhere((x[k] - binwidth <= all_tdata) & (all_tdata <= x[k] + binwidth)).flatten()
                arg_lens[k] = len(args)
            binpops[i, j] = np.median(arg_lens)
        
    fig, ax = plt.subplots()
    
    for i in range(len(binwidths)):
        ax.plot(zs, binpops[:, i], label=f'$\pm${binwidths[i]} Days')
    ax.legend()
    ax.set(xlabel='Target SNe Redshift', ylabel='Median Reference Curve Bin Population')
    
    fig.savefig(file_location + "Binpop_vs_binwidth.png", dpi=400, bbox_inches='tight')
    fig.savefig(file_location + "Binpop_vs_binwidth.pdf", dpi=400, bbox_inches='tight')


def plot_averaged_widths_vs_stretch(prepeak=False):
    ''' Plots the width vs redshift for the z-band fit data. 
    Parameters
    ----------
    fitband : str
    average : bool
        If True, we'll plot the average of two bands of width data
    secondband : str
        If average == True, this is the second band that we'll use in the averaging. 
    prepeak : bool
        If true, will plot the fits that require pre-peak lightcurve data. 
    '''
    suffix = "_PREPEAK" if prepeak else ""
    file_location = "Images/Pre-peak Images/" if prepeak else "Images/"
    
    # load in data from our fit band
    with open(filenames['g'] + '_pickled_data' + suffix, 'rb') as file:
        data1 = pickle.load(file)
    with open(filenames['r'] + '_pickled_data' + suffix, 'rb') as file:
        data2 = pickle.load(file)
    with open(filenames['i'] + '_pickled_data' + suffix, 'rb') as file:
        data3 = pickle.load(file)
    with open(filenames['z'] + '_pickled_data' + suffix, 'rb') as file:
        data4 = pickle.load(file)
        
    # cids1, cids2, cids3, cids4 = data1["CID"].to_numpy(), data2["CID"].to_numpy(), data3["CID"].to_numpy(), data4["CID"].to_numpy()
    allSNe = {}
    data_list = [data1, data2, data3, data4]
    for i, data in enumerate(data_list):
        for j, cid in enumerate(data["CID"].to_numpy()):
            if cid not in allSNe:
                if data["Width_err"][j] < data["Width"][j]:
                    allSNe[cid] = [1, data["z"][j], data["Width"][j], data["Width_err"][j]**2]
            else:
                if data["Width_err"][j] < data["Width"][j]:
                    allSNe[cid] = [allSNe[cid][0] + 1, data["z"][j], allSNe[cid][2] + data["Width"][j], allSNe[cid][3] + data["Width_err"][j]**2]
    
    widths = np.array([allSNe[cid][2] / allSNe[cid][0] for cid in allSNe])
    zs = np.array([allSNe[cid][1] for cid in allSNe])
    werr = np.array([np.sqrt(allSNe[cid][3]) / allSNe[cid][0] for cid in allSNe])
    # colours = np.array([allSNe[cid][0] for cid in allSNe])
    colours = np.array([Methods.sn.loc[Methods.sn['CID'] == cid]['x1'] for cid in allSNe])
    colours = colours.flatten()
    print(colours)
    
    
    import matplotlib as mpl
    # from matplotlib.colors import LinearSegmentedColormap, ListedColormap
    # cmap = ListedColormap(["#f98e08", "#a1a418", "#53a45b", "#1e998a"]) 
    cmap_choice = 'viridis'
    cmap = mpl.cm.get_cmap(cmap_choice)
    alpha = 0.45
    colours2 = []
    maxval = max(colours)
    diff = max(colours) - min(colours)
    def map_0_2_1(val):
        return (val - min(colours)) / diff
    # for i in range(len(colours)):
    #     # print(colours[i] / maxval)
    #     colours2.append(cmap((colours[i] - 1) / maxval))
    colours2 = [cmap(map_0_2_1(colours[i])) for i in range(len(colours))]
        
    # find the best fit (chi square minimisation by default) of the width vs 1+z curve
    best_fit, cov = opt.curve_fit(power_1z, 1 + zs, widths, p0=(1,),
                                  sigma=werr, absolute_sigma=True)
    best_fit = best_fit[0]  # this is our b parameter
    err = np.sqrt(np.diag(cov))[0]  # covariance matrix error
    print(f"b = {best_fit} \pm {err}")
    red_chisquare = sum([(((1 + zs[i])**best_fit - widths[i]) / werr[i])**2 for i in range(len(widths))]) / (len(widths) - 1)
    print(f"all band average, prepeak={prepeak}, reduced chi square = {red_chisquare:.3f}")

    
    # now calculate the binned data for visualisation purposes
    binX, binY, binYerr = Methods.bin_data(1 + zs, widths, yerr=werr, n=50)

    fig, ax = plt.subplots(figsize=(9, 5))
    # plot the raw width data
    # width_plot = ax.errorbar(1 + zs, widths, yerr=werr,
    #             fmt='.', color=colours, alpha=0.3, label='Data', rasterized=True)
    width_plot = ax.errorbar(1 + zs, widths, yerr=werr,
                fmt=',', ecolor=colours2, alpha=alpha/2, label='Data', rasterized=True)
    scatter_plot = ax.scatter(1 + zs, widths, c=colours, cmap=cmap_choice, s=2, alpha=alpha, rasterized=True)
    # legend1 = ax.legend(*width_plot.legend_elements(), title="Averaged Bands")
    # now plot the errorbars on top
    ax.errorbar(binX, binY, yerr=binYerr, fmt='.', c='tab:red',
                label='Binned Data', rasterized=True)
    # plot a 1 + z line
    line = np.linspace(1, 1.01 + max(zs), 5)
    ax.plot(line, line, c='k', ls=':', label='$1 + z$')
    # now plot the line of best fit
    rbest = f"{best_fit:.3f}"
    rerr = f"{err:.3f}"
    ax.plot(line, power_1z(line, best_fit), c='tab:blue',
            ls='--', label='$(1 + z)^{'+rbest+'\pm'+rerr+'}$')
    # now plot the uncertainty in the line of best fit
    for sign in [1, -1]:
        ax.plot(line, power_1z(line, best_fit + sign * err),
                c='tab:blue', ls=':', alpha=0.7)
    ax.set_ylim(0.5, 3)
    ax.set_xlim(1, 1.01 + max(zs))
    ax.set_xlabel(r"$1 + z$")
    ax.set_ylabel("Lightcurve width $w$")
    ax.legend(loc='upper left')
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    colourbar = fig.colorbar(scatter_plot, label='SALT3 x1 Stretch', pad=0, aspect=40)
    colourbar.solids.set(alpha=1)
    fig.savefig(f"{file_location}AllAveWidths-vs-1+z-first{len(zs)}{suffix}-SALTStretch.png", dpi=400, bbox_inches='tight')
    fig.savefig(f"{file_location}AllAveWidths-vs-1+z-first{len(zs)}{suffix}-SALTStretch.pdf", dpi=400, bbox_inches='tight')
        
    
    
def stretch_vs_salt_stretch(prepeak=False):
    ''' Plots the width vs redshift for the z-band fit data. 
    Parameters
    ----------
    fitband : str
    average : bool
        If True, we'll plot the average of two bands of width data
    secondband : str
        If average == True, this is the second band that we'll use in the averaging. 
    prepeak : bool
        If true, will plot the fits that require pre-peak lightcurve data. 
    '''
    suffix = "_PREPEAK" if prepeak else ""
    file_location = "Images/Pre-peak Images/" if prepeak else "Images/"
    
    # load in data from our fit band
    with open(filenames['g'] + '_pickled_data' + suffix, 'rb') as file:
        data1 = pickle.load(file)
    with open(filenames['r'] + '_pickled_data' + suffix, 'rb') as file:
        data2 = pickle.load(file)
    with open(filenames['i'] + '_pickled_data' + suffix, 'rb') as file:
        data3 = pickle.load(file)
    with open(filenames['z'] + '_pickled_data' + suffix, 'rb') as file:
        data4 = pickle.load(file)
        
    # cids1, cids2, cids3, cids4 = data1["CID"].to_numpy(), data2["CID"].to_numpy(), data3["CID"].to_numpy(), data4["CID"].to_numpy()
    allSNe = {}
    data_list = [data1, data2, data3, data4]
    for i, data in enumerate(data_list):
        for j, cid in enumerate(data["CID"].to_numpy()):
            if cid not in allSNe:
                if data["Width_err"][j] < data["Width"][j]:
                    allSNe[cid] = [1, data["z"][j], data["Width"][j], data["Width_err"][j]**2]
            else:
                if data["Width_err"][j] < data["Width"][j]:
                    allSNe[cid] = [allSNe[cid][0] + 1, data["z"][j], allSNe[cid][2] + data["Width"][j], allSNe[cid][3] + data["Width_err"][j]**2]
    
    widths = np.array([allSNe[cid][2] / allSNe[cid][0] for cid in allSNe])
    zs = np.array([allSNe[cid][1] for cid in allSNe])
    werr = np.array([np.sqrt(allSNe[cid][3]) / allSNe[cid][0] for cid in allSNe])
    # colours = np.array([allSNe[cid][0] for cid in allSNe])
    colours = np.array([Methods.sn.loc[Methods.sn['CID'] == cid]['x1'] for cid in allSNe])
    colours = colours.flatten()
    # print(colours)
    
    stretches = widths / (1 + zs)
    stretches_err = werr / (1 + zs)
    def salt_stretch(x1):
        return 0.98 + 0.091 * x1 + 0.003 * x1**2 - 0.00075 * x1**3
    SALT_stretches = salt_stretch(colours)
    
    fig, ax = plt.subplots()
    
    ax.errorbar(1 + zs, stretches - SALT_stretches, yerr=stretches_err, fmt='.', alpha=0.5)
    ax.axhline(0, ls='--', c='k')
    
    def redshift_drift(z):
        deltaz = 1 / ((1 + z)**-2.8 / 0.87 + 1)
        return deltaz * 0.37 + (1 - deltaz) * (0.51 * 0.37 + (1 - 0.51) * -1.22)
    
    fig, ax = plt.subplots()
    
    ax.errorbar(1 + zs, stretches - salt_stretch(redshift_drift(zs)), yerr=stretches_err, fmt='.', alpha=0.5, c='tab:grey')
    ax.axhline(0, ls='--', c='k')
    
def redshift_drift_plots():
    '''Code written by Tamara, edited by Ryan'''
    import matplotlib
    a=0.51
    mu1=0.37
    mu2=-1.22
    sigma1=0.61
    sigma2=0.56
    K=0.87
    
    def stretch(x1):
    	return 0.98 + 0.091*x1 + 0.003*x1**2 - 0.00075*x1**3
    
    def rect(x, y, w, h, c, ax):
        polygon = plt.Rectangle((x,y),w,h,color=c)
        ax.add_patch(polygon)
    def rainbow_fill(X, Y, ax, cmap=plt.get_cmap("jet")):
        ax.plot(X, Y, lw=0)  # Plot so the axes scale correctly
        dx = X[1]-X[0]
        N = float(X.size)
        for n, (x,y) in enumerate(zip(X,Y)):
            color = cmap(n/N)
            rect(x, 1, dx, y-1, color, ax)
    
    #####################################################
    # Test the stretch distribution of our data
    #####################################################
    #df = pd.read_csv('../data/data_for_DESI/good/hubble_diagram.txt', delim_whitespace=True,comment='#') #, names=['CID', 'IDSURVEY', 'zHD', 'zHEL', 'MU', 'MUERR', 'MUERR_VPEC', 'MUERR_SYS'])
    df = pd.read_csv('DES-SN5YR_HD+MetaData.csv')
    df.sort_values('zHD',inplace=True)
    df=df[df['IDSURVEY']==10] # Just choose DES
    x1dat=df['x1'].to_numpy()
    zdat =df['zHD'].to_numpy()
    mBdat=df['mB'].to_numpy()
    cdat=df['c'].to_numpy()
    fit1 = np.polyfit(zdat,x1dat,1)
    print(fit1)
    
    scatter_s = 5
    fig, ax = plt.subplots(figsize=(5, 2.5))
    ax.scatter(zdat, x1dat, c=cdat, alpha=0.8, s=scatter_s, rasterized=True, cmap='viridis_r') 
    ax.plot(zdat,fit1[0]*zdat+fit1[1],color='black',label='Best fit: $x_1={:.4f}z+{:.2f}$'.format(fit1[0],fit1[1]))
    ax.set(xlabel='Redshift, $z$', ylabel='$x_1$')
    # ax.legend(frameon=False)
    ax.legend(framealpha=0.3)
    cmappable = matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(min(cdat), max(cdat)), cmap='viridis_r')
    colourbar = fig.colorbar(cmappable, ax=ax, label='SN Colour', pad=0, aspect=30)
    fig.savefig('Images/x1_vs_z_DES-SN5YR.png', bbox_inches='tight')
    fig.savefig('Images/x1_vs_z_DES-SN5YR.pdf', bbox_inches='tight')
    
    sdat = stretch(x1dat)
    fit2 = np.polyfit(zdat,sdat,1)
    print(fit2)
    fig, ax = plt.subplots(figsize=(5.5, 3))
    ax2 = ax.twinx()
    ax.scatter(zdat, sdat, c=cdat, alpha=0.8, s=scatter_s, rasterized=True, cmap='viridis_r') 
    ax.plot(zdat, fit2[0]*zdat+fit2[1], color='k', label='Best fit: $s={:.4f}z+{:.2f}$'.format(fit2[0],fit2[1]))
    ax2.scatter(zdat, x1dat, alpha=0, rasterized=True) 
    ax2.yaxis.set_tick_params(labelsize=9)
    ax.set(xlabel='Redshift, $z$', ylabel='Stretch')
    # ax.legend(frameon=False)
    ax.legend(framealpha=0.3)
    cmappable = matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(min(cdat), max(cdat)), cmap='viridis_r')
    colourbar = fig.colorbar(cmappable, ax=ax, label='SN Colour', aspect=30, pad=0.08)
    fig.savefig('Images/s_vs_z_DES-SN5YR.png', bbox_inches='tight')
    fig.savefig('Images/s_vs_z_DES-SN5YR.pdf', bbox_inches='tight')
    
    ####################################################
    # Choose which redshifts to use for the rest of the analysis
    ####################################################
    zs = np.arange(0,1.41,0.1)
    x1s = np.arange(-3,3,0.1)
    # zs = zdat
    # x1s = x1dat
    
    #####################################################
    ##### Investgating the populations and how x1 drifts.
    #####################################################
    fig, ax = plt.subplots(figsize=(5, 4))
    norm1 = (1/(sigma1*np.sqrt(2*np.pi)))*np.exp((-(x1s-mu1)**2)/(2*sigma1**2))
    norm2 = (1/(sigma2*np.sqrt(2*np.pi)))*np.exp((-(x1s-mu2)**2)/(2*sigma2**2))
    #x1=mu1-np.sqrt(np.alog(norm)*(2*sigma1**2)) # Trying to create a gaussian about the mean.  Work in progress!
    ax.plot(x1s, norm1, '--', label='main', color='k')
    ax.plot(x1s, norm2, '-.', label='secondary', color='grey')
    
    X1_prob = np.zeros([len(zs),len(x1s)])
    x1_mean = np.zeros(len(zs))
    plt.get_cmap('rainbow')
    x = np.linspace(0, 2*np.pi, len(zs))
    colors = plt.cm.jet(np.linspace(0,1,len(zs)))
    for i, z in enumerate(zs):	
        delta=(1/K*(1+z)**(-2.8) +1)**(-1)	
        X1_prob[i,:] = delta*norm1 + (1-delta)*(a*norm1 + (1-a)*norm2)
        x1_mean[i] = np.sum(X1_prob[i,:]*x1s)/np.sum(X1_prob[i,:])
    	# ax.plot(x1s, X1_prob[i,:], label='$z={:.1f}$'.format(z), color=colors[i])
        ax.plot(x1s, X1_prob[i, :], color=colors[i])
        ax.plot([x1_mean[i], x1_mean[i]], [0.0, 1.0], ':', color=colors[i])
    ax.set(xlabel='$x_1$', ylabel='$x_1$ distribution', ylim=(0, 1))
    ax.legend(frameon=False)
    
    zs = np.arange(0,1.41,0.001)
    cmappable = matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(0, max(zs)), cmap='jet')
    colourbar = fig.colorbar(cmappable, ax=ax, label='Redshift, $z$', pad=0, aspect=40)
    fig.savefig('Images/x1drift.png', bbox_inches='tight')
    fig.savefig('Images/x1drift.pdf', bbox_inches='tight')
    
    
    X1_prob = np.zeros([len(zs),len(x1s)])
    x1_mean = np.zeros(len(zs))
    for i, z in enumerate(zs):	
        delta=(1/K*(1+z)**(-2.8) +1)**(-1)	
        X1_prob[i,:] = delta*norm1 + (1-delta)*(a*norm1 + (1-a)*norm2)
        x1_mean[i] = np.sum(X1_prob[i,:]*x1s)/np.sum(X1_prob[i,:])
    fig, ax = plt.subplots()
    ax.plot(zs, x1_mean, '-')
    ax.set(ylim=(-0.4,0.4), xlabel='Redshift, $z$', ylabel='Mean $x_1$', xscale='log')
    fig.savefig('Images/x1drift_vs_z.png', bbox_inches='tight')
    fig.savefig('Images/x1drift_vs_z.pdf', bbox_inches='tight')
    
    
    #####################################################
    # Convert x1 to stretch and plot
    #####################################################
    # X1 is easy to convert to stretch as there are published conversions. 
    # Stretch is directly related to the width of the light curve 
    # (an s=1.1 SN is 10% faster than an s=1 SN, etc.).
    
    s = stretch(x1_mean) #0.98 + 0.091*x1_mean + 0.003*x1_mean**2 - 0.00075*x1_mean**3
    fig, ax = plt.subplots(figsize=(5, 1.56))
    rainbow_fill(zs, s, ax)
    ax.plot(zs, s, '-', color='k', linewidth=2)
    ax.set(xlabel='Redshift, $z$', ylabel='Mean Stretch')
    fig.savefig('Images/x1drift-stretch_drift_vs_z.png', bbox_inches='tight')
    fig.savefig('Images/x1drift-stretch_drift_vs_z.pdf', bbox_inches='tight')
    
    # The magnitude (and thus the distance modulus) 
    # are related to x1 by the term $\Delta m = \alpha x_1$.  
    # See Eq. 1 of key paper.  alpha = 0.161 # ± 0.001
    
    ######################################################
    # Now let's mock that up and see what the slope is. 
    ######################################################
    timedil = 1+zs
    timedil_drift = (1+zs)*s #+(1-s[0])
    fit = np.polyfit(1+zs,timedil_drift,1)
    bs=np.arange(0.98,1.02,0.001)
    chi2 =  np.zeros(len(bs))
    uncert= 0.14 #arbirtrary uncertainty designed to make chi2~1
    for i,b in enumerate(bs):
    	timdil_theory = (1+zs)**b
    	chi2[i] = np.sum((timedil_drift-timdil_theory)**2/uncert**2)
    	print(i,chi2[i])
    ibest = np.argmin(chi2)
    bbest = bs[ibest]
    print(ibest,bbest)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(1+zs, timedil, ':', c='tab:blue', label='Time dilation only: $\Delta t=(1+z)$')
    ax.plot(1+zs, 1.03 * (1 + zs) - 0.05, '--', c='tab:red', label='with stretch drift: $\Delta t=1.03(1+z)-0.05$') ## HARDCODED
    ax.plot(1+zs, np.ones(len(zs)), '-', c='k', label='no time dilation: $\Delta t = 0$')
    #ax.plot(1+zs,timedil_drift,':',label='with stretch drift: td$={:.2f}(1+z) +{:.2f}$ or td$=(1+z)^{:.4f}$'.format(fit[0],fit[1],bbest))
    # ax.plot(1+zs, timedil_drift, '--', label='With stretch drift: $\Delta t \sim (1+z)^{{{:.3f}}}$'.format(bbest)) # WARNING HARDCODED RESULT
    
    ax.legend(frameon=False)
    ax.set(xlabel='$1+z$', ylabel='Light Curve Width, $w$')
    fig.savefig('Images/x1drift-td.png',bbox_inches='tight')
    fig.savefig('Images/x1drift-td.pdf',bbox_inches='tight')
    
def lightcurve_stretch_vs_z(band='i'):
    import matplotlib
    fig, ax = plt.subplots(figsize=(8, 6))
    Zs = Methods.Zs
    cmap = matplotlib.cm.get_cmap('plasma', len(Zs))
    cmappable = matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(min(Zs), max(Zs)), cmap='plasma')
    
    
    
    for i, cid in enumerate(Methods.CIDs):
        try:
            t, data, _ = Methods.get_SN_data(cid, band)
            tsim, sim, _ = Methods.get_SN_sim(cid, band)
            ax.scatter(t, data / max(sim), color=cmap(Zs[i] / max(Zs)), s=1)
            ax.plot(tsim, sim / max(sim), color=cmap(Zs[i] / max(Zs)), alpha=0.5)
        except:
            continue
            
    colourbar = fig.colorbar(cmappable, ax=ax, label='Redshift $z$', pad=0, aspect=30)
    ax.set(xlabel=r'Time Since Peak Brightness, $t - t_{peak}$ (days)', ylabel='Normalised Flux', ylim=(-0.2, 1.2), xlim=(-40, 100))
    ax.grid()
    fig.savefig(f'Images/Lightcurves_vs_Redshift_{band}band.png', dpi=400, bbox_inches='tight')
    

def main():
    ## --- First, plot the width vs redshift for all bands --- ##
    for band in ['g', 'r', 'i', 'z']:
        plot_width_vs_z(band)
        plot_width_vs_z(band, prepeak=True) # now plot it again for the required prepeak data case
        plot_width_vs_z(band, nonderedshifted=True) # now plot it again for the nonderedshifted data
    
    # ## --- Also good to plot all of the widths on the same figure --- ##
    plot_all_widths()
    plot_all_widths(prepeak=True)
    plot_all_widths(nonderedshifted=True)
    
    ## -- Now, plot the average of the z band and i band data against (1 + z) -- ##
    plot_width_vs_z('z', average=True, secondband='i')
    plot_width_vs_z('z', average=True, secondband='i', prepeak=True)
    
    
    ## -- For good measure, plot how well the z band and i band widths agree -- ##
    for band in ['z', 'r', 'g']:
        compare_widths('i', band)
        compare_widths('i', band, prepeak=True)
    
    compare_all_widths()    # good to compare all widths on the same figure!
    compare_all_widths(prepeak=True)
    
    # # -- Now, plot the dispersion in the reference curves over all of the lightcurves for each method -- ##
    # for method in ['power', 'linear']:
    #     plot_dispersion_vs_scaling(method=method)
    for band in ['g', 'r', 'i', 'z']:
        plot_dispersion_vs_scaling(band, method='power')
    plot_all_dispersion_vs_scaling()

    ## -- Finally, plot the available data for an individual SNe to see how the photometry evolves in colour -- ##
    index = np.where(CIDs == 1303317)[0][0]  # this seems to be a nice curve!
    plot_lightcurve_bands(CIDs[index], Zs[index], save=True)

    ## -- Plot some lightcurve fits with their reference pops -- ##
    bands = ['i', 'r', 'z']
    
    # create a list filled with (cid, z) for 3 different SNe. Values chosen because they look good!
    cid_zs = [(CIDs[index], Zs[index]) for index in [10, np.argwhere(
        Zs < 0.3).flatten()[0], np.argwhere(Zs > 0.8).flatten()[0]]]
    for i, (cid, z) in enumerate(cid_zs):
        plot_stretch(cid, z, bands[i])
    
    # # the below 2 lines were to identify a bug in the flux error cutting
    # # ind = np.where(CIDs == 1479069)[0][0] #
    # # plot_stretch(CIDs[ind], Zs[ind], 'r')
        
    ## --- Plot the reference creation for the medium redshift example above --- ##
    index = 0
    plot_reference_creation(cid_zs[index][0], cid_zs[index][1], bands[index])
        
    # --- Now plot the reference population for each SNe w.r.t. wiggle --- ##
    for band in ['g', 'r', 'i', 'z']:
        plot_reference_pops(band)
    
    
    ## --- Plot the whole dataset for good measure --- ##
    plot_flux_vs_error()
    
    ## --- Plot the spectrum showing the effect of redshifting --- ##
    plot_2001V_spec()
    
    ## --- Average all of the widths across the bands to get a final time dilation result --- ##
    plot_averaged_widths()
    plot_averaged_widths(prepeak=True)
    
    ## --- Plot the types of error vs redshift --- ##
    plot_err_floor_vs_MC()
    # for i in [85, 97, 52, 143]:  # 0.15, ~0.65, 0.8, 1.01
    #     if i == 85:
    #         plot_stretch(CIDs[i], Zs[i], 'g', width_frac=2**-3)
    #     else:
    #         plot_stretch(CIDs[i], Zs[i], 'z', width_frac=2**-3)
    
    plot_w_vs_m()
    plot_binpop_vs_binwidth(N=20)
    
    plot_averaged_widths_vs_stretch(prepeak=False)
    
    # stretch_vs_salt_stretch()
    redshift_drift_plots()
    
    for band in ['g', 'r', 'i', 'z']:
        lightcurve_stretch_vs_z(band=band)


if __name__ == "__main__":
    main()
