#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
from scipy import linalg  # numpy linalg plus extras
#import statsmodels.api as sm
from numpy import *
#from pycurrents.system import Bunch
from scipy import ndimage
from scipy import interpolate
from scipy import constants
import xarray as xr
import pandas as pd
from datetime import datetime


# In[37]:


get_ipython().run_line_magic('pinfo', 'interpolate.interp1d')


# In[29]:


def profile_interp(var,z_orig,z_interp,method='linear',out_of_bounds='NaN'):
    """ Wrapper method. Use 1-D interpolation method of choice to evaluate 1-D 'var' at value/values 'z_interp'.
    Args:
        var=array-like
        method: 'linear'  for linear interpolation
                'nearest' for nearest-neighbor interpolation
                               (in which case 'nearest' and 'extrap' as args for 'out_of_bounds' are identical)
                'cubic'   for spline interpolation of the 3rd order
        out_of_bounds: 'NaN'     to return NaN for values above or below range of 'z_orig'
                       'nearest' to extrapolate using the uppermost/lowermost value of 'data'
                       'extrap'  to extrapolate using cubic interp
    
    """
    z_orig = z_orig[~isnan(z_orig)]
    var= var[~isnan(var)]
    #assert(all(diff(z_orig) > 0))
    if len(z_orig) > len(var) or len(var) > len(z_orig): return NaN
    if len(z_orig) <= 2 or len(var) <= 2: return NaN
    
    if out_of_bounds == 'NaN':
        interpolant = interpolate.interp1d(z_orig,var,kind=method,bounds_error=False,fill_value=NaN)
    elif out_of_bounds == 'nearest':
        interpolant = interpolate.interp1d(z_orig,var,kind=method,bounds_error=False,fill_value=(var[0],var[-1]))
    elif out_of_bounds == 'extrap':
        interpolant = interpolate.interp1d(z_orig,var,kind=method,bounds_error=False,fill_value='extrapolate')
    else:
        raise ValueError('Extrapolation method must be NaN, nearest, or cubic.')
    result = interpolant(z_interp)

    if result.size == 1: return result.item()
    else:                return result


# In[14]:


def vert_prof_eval(var,z_coor,z_or_z_range,interp_method='linear',extrap='NaN',
                  avg_method='interp',avg_spacing=0.1,avg_nan_tolerance=0.0,verbose_warn=True,verbose_error=True):
    """ Compute interpolated value at a depth/depths OR average value within range of depths for a vertical profile.

    NOTE: requires positive, monotonically-increasing vector of depths or pressures. 

    Args:
        var= array-like, values to be interpolated
        z_or_z_range: three options:
            [a] single z value (to interpolate single value) [NOTE: array(scalar) will return a scalar, not an array)]
            [b] array or list of z values (to interpolate multiple values)
            [c] 1x2 tuple of (ztop,zbottom) where ztop < zbottom (to compute average value within range)
        z_coor: 'depth' or 'pres' (remember to give z_or_z_range in meters or decibars accordingly)
        interp_method: evaluate profile data using 'linear', 'nearest', or 'cubic' interpolation ('linear' recommended)
        extrap: extrapolate above/below z range via 'NaN', 'nearest', or 'extrap'
        avg_method: 'simple' for simply averaging all values found within the z_range
                    'interp' for interpolating data to regular spacing (determined by 'spacing') and then averaging
        avg_spacing: approximate spacing in meters or decibars of interpolated values for averaging
                     (relevant only if z_or_z_range is a tuple ([c]) and avg_method == 'interp')
                     (approximate in order to keep spacing perfectly even between upper and lower z values)
        avg_nan_tolerance: print error and return NaN if NaN fraction of original or interpolated data values
                                        in z range is > nan_tolerance
                           (float between 0.0 and 1.0)
                           (note: relevant only if z_or_z_range is a tuple ([c]))
        verbose_warn: print warnings
        verbose_error: print fatal errors (recommend keeping this True)

    Returns:
        None if error encountered
        computed value/values (which can be NaN) if successful

    """

    # evaluate data at z value/values
    if not isinstance(z_or_z_range,tuple):
        #var= array(var,dtype=float) specify dtype if unsupported input type
        return profile_interp(var,z_coor,z_or_z_range,
                              method=interp_method,out_of_bounds=extrap)

    # compute average within range (tuple) of z values
    else:
        if avg_method == 'simple':
            z_match = logical_and(z_coor <= z_or_z_range[1],
                                  z_coor >= z_or_z_range[0])
            if sum(z_match) == 0:
                if verbose_warn: print('Warning : no data within given depth range.')
                return NaN
            else:
                if sum(isnan(var[z_match])) / sum(z_match) > avg_nan_tolerance:
                    if verbose_warn: print('Warning : too many NaNs in given depth range.')
                    return NaN
                else:
                    return nanmean(var[z_match])
        elif avg_method == 'interp':
            z_eval = linspace(z_or_z_range[0],z_or_z_range[1],
                              int(ceil((z_or_z_range[1] - z_or_z_range[0]) / avg_spacing)))
            data_to_avg = profile_interp(var,z_coor,
                                         z_eval,method=interp_method,out_of_bounds=extrap)
            if isinstance(data_to_avg,float):
                if isnan(data_to_avg):
                    if verbose_warn: print('Warning  '
                                           'too little data; unable to interpolate.')
                    return NaN
            elif sum(isnan(data_to_avg)) / len(data_to_avg) > avg_nan_tolerance:
                if verbose_warn: print('Warning too many NaNs in given depth range.')
                return NaN
            else:
                return nanmean(data_to_avg)


# In[12]:


def vert_prof_even_spacing(var,z_coor,spacing=0.1,interp_method='linear',extrap='NaN',
                           top=0.0,bottom='bottom',verbose_error=True):
    """ Interpolates vertical profile to even spacing. Helpful wrapper function for vert_prof_eval().

    Args:
        spacing: in meters or decibars (note: will start/end spacing inside range, e.g. given spacing of 0.25 and
                                        z-values from 5.1 to 1499.9, will return inclusive array from 5.25 to 1499.75;
                                        that said, will start/end spacing at given bounds if they line up with spacing)
        interp_method: see vert_prof_eval()
        extrap: see vert_prof_eval()
        top: <<scalar>> to start at given level or 'top' to start at uppermost measured level
        bottom: <<scalar>> to end at given level or 'bottom' to end at bottommost measured level
        verbose_error: print fatal errors (recommend keeping this True)

    Returns:
        z_vec, data_vec

    """

    if top == 'top':
        top = z_coor[0]
    if bottom == 'bottom':
        bottom =z_coor[-1]
    z_vec = xr.DataArray(arange(0.0, bottom+spacing, spacing))
    z_vec= z_vec.where(z_vec >= top, z_vec <= bottom)
    z_vec= array(z_vec, dtype= float)
    data_vec = vert_prof_eval(var,z_coor,z_vec,interp_method=interp_method,extrap=extrap,
                             verbose_error=verbose_error)
    return z_vec, data_vec
        


# In[6]:


def depth_at_which(var,z_coor,value_attained,method='actual',top='top',bottom='bottom',
                   interp_initial_spacing=1.0,interp_final_spacing=0.01,verbose_warn=True,verbose_error=True):
    """ Estimate depth at which a given value is attained (intersected) in a vertical profile.

    Important notes on function behavior:
        Note that search direction is downwards from <<top>> pressure/depth level to <<bottom>> level.
        If parameter value at <<top>> is less than or equal to <<value_attained>>, function will search for first level
           at which <<value_attained>> is exceeded.
        If parameter value at <<top>> exceeds <<value_attained>>, function will search for first level at which
            parameter is less than <<value_attained>>.
        Function can also search for levels of max/min value between <<top>> and <<bottom>>.

    Args:
        var: for mld, sigma_theta
        value_attained: three options for value of <<param_abbrev>> to search for:
            [a] scalar: search for this value
            [b] 'max': search for maximum value
            [c] 'min': search for minimum value
        z_coor: 'depth' or 'pres'
        method: 'actual' to choose measurement level preceding first measured level where value_attained is attained
                            (note that this will underestimate rather than overestimate the level)
                'interp' to use linear interpolation with 'nearest' interpolation to estimate exact level (recommended)
        top: <<scalar>> to start searching at given level or 'top' to start at uppermost measured level
        bottom: <<scalar>> to end searching at given level or 'bottom' to end at bottommost measured level
        interp_initial_spacing: spacing in meters/decibars used for interpolation during initial, coarse search
        interp_final_spacing: spacing in meters/decibars used for interpolation during final, fine search
                              (must be ≤ crit_interp_initial_spacing)
                              (note: these spacing args are only used if 'interp' selected for 'method')
        verbose_warn: print warnings
        verbose_error: print fatal errors (recommend keeping this True)

    Returns:
        level (depth in meters or pressure in decibars) at which <<value_attained>> attained
        NaN if <<value_attained>> is not attained between <<top>> and <<bottom>>
        None if error encountered

    """

    # get search bounds
    if top == 'top':
        top = z_coor[0]
    if bottom == 'bottom':
        bottom = z_coor[-1]

    # determine whether parameter values are increasing or decreasing
    if value_attained != 'max' and value_attained != 'min':
        first_value = vert_prof_eval(var,z_coor,top,interp_method='linear',
                                     extrap='nearest',verbose_warn=True,verbose_error=True)
        if first_value <= value_attained: expect_increasing = True
        else:                             expect_increasing = False

   
    # search for actual measurement levels
    if method == 'actual':
        levels_in_range_mask = logical_and(z_coor >= top,
                                           z_coor <= bottom)
        levels_in_range = z_coor[levels_in_range_mask]
        data_in_range   = var[levels_in_range_mask]

        if value_attained == 'max':
            attained_idx = argmax(data_in_range)
        elif value_attained == 'min':
            attained_idx = argmin(data_in_range)
        else:
            if expect_increasing:
                attained = (data_in_range >= value_attained)
            elif not expect_increasing:
                attained = (data_in_range <= value_attained)

            if sum(attained) == 0:
                return NaN
            else:
                attained_idx = argmax(attained) - 1  # note: np.argmax returns index of first 'True', or 0 if all False (!)
        if attained_idx == -1:
            return NaN
        else:
            return levels_in_range[attained_idx]

    # use interpolation to estimate depth of interest
    
    elif method == 'interp':
        # initial, coarse search for vicinity of depth
        lev_coarse, data_coarse = vert_prof_even_spacing(var,z_coor,
                                                         spacing=interp_initial_spacing,interp_method='linear',
                                                         extrap='nearest',top=top,bottom=bottom,verbose_error=True) 
        
        if value_attained == 'max':
            attained_idx_coarse = argmax(data_coarse)
        elif value_attained == 'min':
            attained_idx_coarse = argmin(data_coarse)
        else:
            if expect_increasing:
                attained = (data_coarse >= value_attained)
            elif not expect_increasing:
                attained = (data_coarse <= value_attained)

            if sum(attained) == 0:
                return NaN
            else:
                attained_idx_coarse = argmax(attained) - 1

        # final, fine search for depth
        if attained_idx_coarse == 0: top_idx_coarse = 0
        else:                        top_idx_coarse = attained_idx_coarse - 1
        if attained_idx_coarse == len(lev_coarse)-1: bottom_idx_coarse = len(lev_coarse)-1
        else:                                        bottom_idx_coarse = attained_idx_coarse + 1
        lev_fine, data_fine = vert_prof_even_spacing(var,z_coor,
                                                     spacing=interp_final_spacing,interp_method='linear',
                                                     extrap='nearest',top=lev_coarse[top_idx_coarse],
                                                    bottom=lev_coarse[bottom_idx_coarse],verbose_error=True)
        if value_attained == 'max':
            attained_idx_fine = argmax(data_fine)
        elif value_attained == 'min':
            attained_idx_fine = argmin(data_fine)
        else:
            if expect_increasing:
                attained = (data_fine >= value_attained)
            elif not expect_increasing:
                attained = (data_fine <= value_attained)

            if sum(attained) == 0:
                return NaN
            else:
                attained_idx_fine = argmax(attained) - 1

        return lev_fine[attained_idx_fine]

    else:
        if verbose_error: print('Error : check argument passed for method.')


# In[7]:


def calc_mld(data,ref_depth=10,ref_range_method='interp',ref_reject=False,sigma_theta_crit=0.03,crit_method='interp',
        crit_interp_initial_spacing=1.0,crit_interp_final_spacing=0.01,bottom_return='NaN',
        verbose_warn=True,verbose_error=True):
    """ Compute mixed layer depth (MLD) given a vertical profile of sigma-theta (potential density anomaly).
    
    Args:
        data: xarray dataset with 'sigma_theta' as a variable "var"  and variable 'depth' as "z_coor"
                      (note that a positive, monotonically increasing depth vector required, not pressure)
        ref_depth: three options for reference depth(s) in meters:
            [a] single scalar value at which sigma_theta evaluated using linear interp with 'nearest' extrapolation
            [b] range of values expressed as tuple of scalars: (upper,lower), where lower > upper
            [c] 'shallowest' (string), indicating the shallowest available measurement
        ref_range_method: if [b] above, calculate average in range using 'simple' or 'interp'? (see vert_prof_eval())
                         (for 'interp', linear interpolation with 'nearest' extrapolation used before averaging)
                         (if [b] not selected, this arg is ignored)
        ref_reject: False (default) or True (to return 'NaN' if ref_depth is [a] or [b] above and shallowest measurement
                    is above value for [a] or upper value for [b]
        sigma_theta_crit: density criteria in kg/m3 as scalar
        crit_method: how to select the MLD using the given criteria?
            'actual' to choose measurement depth preceding first measured depth where sigma_theta_crit is exceeded
                     (probably better to slightly underestimate MLD than overestimate it)
            'interp' to use linear interpolation with 'nearest' interpolation to estimate exact MLD (recommended)
        crit_interp_initial_spacing: spacing in meters used for interpolation during initial, coarse MLD search
        crit_interp_final_spacing: spacing in meters used for interpolation during final, fine MLD search
                                  (must be ≤ crit_interp_initial_spacing)
                                  (note: these spacing args are only used if 'interp' selected for 'crit_method')
        bottom_return: what to return if MLD not reached by bottom of profile
                      (note: warning will be printed if verbose_warn is True)
            'bottom' to return deepest measurement depth
            'NaN' to return NaN
        verbose_warn: print warnings
        verbose_error: print fatal errors (recommend keeping this True)
    Returns:
        MLD in meters if found
        NaN if MLD couldn't be found
        None if error encountered
    Common MLD criteria using sigma_theta:
        de Boyer Montégut et al. 2004 (for global ocean):
             0.03 kg/m3 from value at 10 m
            (authors note that 0.01 kg/m3 had been the 'often standard' criteria)
        Dong et al. 2008 (for Southern Ocean):
             0.03 kg/m3 (or temp criterion) from "near surface" value
            (authors say "0-20 m" or "20 m" but don't specify which to use, or whether to use average value)
        Wong and Riser 2011 (for under-ice Argo profiles off E. Antarctica):
             0.05 kg/m3 from the shallowest measurement
    """
    
    sigma_theta= data['Sigma_theta'][:]
    sigma_theta=sigma_theta[::-1]
    pressure= data['Pressure'][:]
    pressure=pressure[::-1]
    
    #if not all(diff(pressure) > 0):
        #pressure=mask_nonincreasing(pressure)
    if verbose_warn:
        if any(isnan(sigma_theta.any()) or any(isnan(pressure.any()))):
            print('Warning: sigma-theta or depth vector contains NaNs.')

        if ref_depth == 'shallowest':
            if pressure[0] >= 20: print('Warning : shallowest measurement is 20 m or deeper.')
        elif not isinstance(ref_depth,tuple):
            if pressure[0] > ref_depth:
                if not ref_reject: print('Warning : '
                                         'reference depth is above shallowest measurement.')
                else:              return NaN
        elif not ref_reject:
            if pressure[0] > ref_depth[1]:
                if not ref_reject: print('Warning : '
                                         'reference depth range is above shallowest measurement.')
                else:              return NaN

    if ref_depth == 'shallowest':
        rho_mld = sigma_theta_crit + sigma_theta[0]
    elif not isinstance(ref_depth,tuple):
        rho_mld = sigma_theta_crit + vert_prof_eval(sigma_theta,pressure,ref_depth,
                                                    interp_method='linear',extrap='nearest',verbose_warn=True,
                                                    verbose_error=True)
    else:
        rho_mld = sigma_theta_crit + vert_prof_eval(sigma_theta,pressure, ref_depth,
                                                    interp_method='linear',extrap='nearest',avg_method=ref_range_method,
                                                    avg_spacing=0.1,verbose_warn=True,verbose_error=True)
               
    mld_found = depth_at_which(sigma_theta,pressure,rho_mld, method=crit_method,
                               top=0.0,bottom='bottom',interp_initial_spacing=crit_interp_initial_spacing,
                               interp_final_spacing=crit_interp_final_spacing,verbose_warn=True,verbose_error=True)

    if mld_found == None:
        if verbose_error: print('Error : unexpected error encountered at end of function.')
        return None
    elif isnan(mld_found) and bottom_return == 'bottom':
        return pressure[-1]
    elif isnan(mld_found) and bottom_return == 'NaN':
        return NaN
    else:
        return mld_found


# In[ ]:





# In[ ]:




