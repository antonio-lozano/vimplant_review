'''
Created on Wed Jul 13 08:20:28 2021

@author: van Hoof & Lozano


loss_func.get_yield
loss_func.DC
loss_func.hellinger_distance
loss_func.makeGaussian


'''

import numpy as np
from scipy.special import rel_entr
import math
import matplotlib.pyplot as plt
import sys
from scipy.spatial import KDTree

def hellinger_distance(p, q):
    """
    Computes the Hellinger distance between two probability distributions.
    
    Args:
        p (array-like): A probability distribution.
        q (array-like): Another probability distribution.
        
    Returns:
        float: The Hellinger distance between p and q.
    """
    p = np.asarray(p)
    q = np.asarray(q)
    return np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / np.sqrt(2)

## loss functions
def KL(a, b):
    a = np.asarray(a, dtype=np.float)
    b = np.asarray(b, dtype=np.float)
        
    # replace zeros with smallest value
    a[a==0] = sys.float_info.min    
    b[b==0] = sys.float_info.min
    
    x = rel_entr(a, b)

    return np.abs(np.sum(x))

def get_yield(contacts_xyz, goodcoords, empty_score=1.0):
    '''
    
    Optimize for yield between the arrays and the gray tissue
    
    goodcoords = all GM voxels which contain pRF data
    contacts_xyz = electrode grid for which we calc the yield
    
    '''
    # filter good coords
    b1 = np.round(np.transpose(np.array(contacts_xyz)))
    b2 = np.transpose(np.array(goodcoords))
    indices_prf = []
    for i in range(b1.shape[0]):
        tmp = np.where(np.array(b2 == b1[i,:]).all(axis=1))
        if tmp[0].shape[0] != 0:
            indices_prf.append(tmp[0][0])

    contact_hits = len(indices_prf)
    total_points = contacts_xyz.shape[1]
    contacts_yield = (contact_hits / total_points)

    return contacts_yield


def get_yield_fast(contacts_xyz, goodcoords, empty_score=1.0):
    """
    Optimized yield function using KDTree for spatial queries.
    
    Parameters:
    - contacts_xyz: Array of electrode grid points for which we calculate yield.
    - goodcoords: Array of all GM voxels containing pRF data.
    - empty_score: Default score if no points match (1.0 by default).

    Returns:
    - contacts_yield: Proportion of contact points that intersect with goodcoords.
    """
    # Ensure contacts_xyz and goodcoords have compatible shapes
    contacts_rounded = np.round(contacts_xyz.T).astype(int)  # Shape should be (N, 3)
    goodcoords_rounded = np.round(goodcoords.T).astype(int)  # Shape should be (N, 3)

    # Build KDTree for good coordinates
    goodcoords_tree = KDTree(goodcoords_rounded)
    
    # Query KDTree for distances within a threshold
    distances, _ = goodcoords_tree.query(contacts_rounded, distance_upper_bound=1)
    
    # Count matches within threshold
    contact_hits = np.sum(np.isfinite(distances))
    total_points = contacts_rounded.shape[0]  # Use contacts_rounded since it's already transposed
    
    # Calculate yield
    contacts_yield = contact_hits / total_points if total_points > 0 else empty_score

    return contacts_yield

def DC(im1, im2, bin_thresh, empty_score=1.0):
    '''
    Computes the Dice coefficient, a measure of set similarity.
    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size. If not boolean, will be converted.
    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0
        Both are empty (sum eq to zero) = empty_score
        
    Notes
    -----
    The order of inputs for `dice` is irrelevant. The result will be
    identical if `im1` and `im2` are switched.
    '''
    
    im1 = (im1 > bin_thresh).astype(bool)
    im2 = (im2 > bin_thresh).astype(bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return (2. * intersection.sum() / im_sum), im1, im2

def makeGaussian(size, fwhm = 3, center=None):
    ''' Make a square gaussian kernel.

    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    '''
    
    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)
