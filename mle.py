"""
mle.py

This code is part of Optimization of Hardware Parameters on a Real-Time Sound
Localization System paper

It contains some support functions for pso.py

Authors:
    Matheus Farias
    Davi Moreno
    Daniel Filgueiras
    Edna Barros
"""

import numpy as np
from itertools import combinations as comb

propSpeed = 340.29 #propagation speed of signal

def dist (Pi, Pj) : 
    """ 
        Calculate the distance of the arrays Pi and Pj

        Pi, Pj : 1D or 2D ndarrays , each line is a coordinate vector 

        return 2D ndarray with the distance/distances between each line of Pi 
        and Pj 

        Broadcasting allowed
    """
    i = 1 if (Pi.ndim == 2 or Pj.ndim == 2) else None 
    diff = Pi - Pj
    square = diff**2
    dist = np.sqrt(np.sum(square, axis=i, keepdims=True))
    dist = dist.reshape((dist.shape[0], 1)) if (i == None) else dist
    return dist

def arrayMatrix(array):
    """ Generates matrix that is used in MLE calculus for some array

        array: row 2D ndarray with the coordinates of all sensors

               [xi yi zi] is the coordinates of i-th sensor, i in [1..M]
               1th sensor is the reference
               
               array = [[x1 ... xM]
                        [y1 ... yM]
                        [z1 ... zM]].T

        return matrix M

        M = -pseudoInverse([[(xM-x1) ... (xM-x1)]
                            [(yM-y1) ... (yM-y1)]
                            [(zM-z1) ... (yM-y1)]].T)            
    """
    M = array[1:] - array[0]
    M = -np.linalg.pinv(M)
    return M

def mleMatrices (tdoa, array, M):
    """ Generates matrices for MLE (maximum likelihood estimation) calculation
        
        tdoa : column 2D ndarray with the TDOA from some reference sensor
        array : row 2D ndarray with the coordinates of all sensors
               
                [xi yi zi] is the coordinates of i-th sensor, i in [1..M]
                1th sensor is the reference
               
                array = [[x1 ... xM]
                         [y1 ... yM]
                         [z1 ... zM]].T
                
        M : M matrix related with array variable (calculated with 
            arrayMatrix(array))

        
        return matrices M1, M2 that are used in the MLE calculation
        
        Xs = M1 * D1 + M2
        
        Xs => [xs ys zs].T estimate source coordinates column vector
        D1 => estimated distance between the source and the reference sensor
    """
    A = -tdoa * propSpeed
    Baux1 = A**2
    Baux2 = -array[1:]**2 + array[0]**2
    Baux2 = np.sum(Baux2, axis=1, keepdims=True)
    B = 0.5 * (Baux1 + Baux2)    
    M1 = np.dot(M, A)
    M2 = np.dot(M, B)
    return M1, M2

def possibleCoords (M1, M2, referenceSensor):
    """ Calculates the two possible coordinates for the given MLE problem
        
        M1 : M1 matrix from mleMatrices function
        M2 : M2 matrix from mleMatrices function
        referenceSensor : 1D ndarray with coords of reference sensor(1th sensor
                          from array)
        
        return a row 2D ndarray with the two possible coordinates for the given
        MLE multilateration solution
    """
    a = np.sum(M1**2) - 1
    b = 2 * ( np.sum(M1 * M2) - np.sum(M1 * referenceSensor) )
    c = np.sum(M2**2) - 2 * np.sum(M2 * referenceSensor) + \
        np.sum(referenceSensor ** 2)
    coeffs = [a, b, c]
    roots = np.roots(coeffs).reshape(2,1)
    finalCoords = (np.dot(M1, roots.T) + M2).T
    return finalCoords

def mle(tdoa, array, M):
    """ MLE (maximum likelihood estimation) algorithm

        tdoa : column 2D ndarray with the TDOA from some reference sensor        
        array : row 2D ndarray with the coordinates of all sensors
               
                [xi yi zi] is the coordinates of i-th sensor, i in [1..M]
                1th sensor is the reference
               
                array = [[x1 ... xM]
                         [y1 ... yM]
                         [z1 ... zM]].T

        
        return a row 2D ndarray with the two possible coordinates for the given
        MLE problem
    """
    M1, M2 = mleMatrices (tdoa, array, M)
    results = possibleCoords (M1, M2, array[0].reshape((-1,1)))
    return np.real(results)

def mle_hls(tdoa, array, M):
    results = mle(tdoa, array, M)
    #results = np.round(results, decimals=3)
    costs = cost(results, tdoa, array)
    i = np.argmin(costs)
    return results[i]

def h(r1, r2, guess):
    """ Calculate the difference of the distances between r1,guess and r2,guess
        using:
            
        d = dist(r1, guess) - dist(r2, guess)
        
        r1, r2 and guess : 1D or 2D row ndarray with coordinates
        
        return 1D ndarray with the difference of the distances between r1,
        guess and r2, guess
    """
    return dist(r1, guess) - dist(r2, guess)

def cost(guess, tdoa, array):
    """ Calculate the sum of the squares of the loss function of hyperbolic 
        least squares problem
    
        guess : 1D or 2D row ndarray with one or more guesses coordinates
        tdoa : column 2D ndarray with the TDOA from some reference sensor        
        receptorsPositions : row 2D ndarray with the coordinates of all 
                             receptors       
        
        return 2D ndarray of shape (number_of_guesses, 1) with the costs 
        related to all guesses
        
        loss_ij = tdoa_ij * soundSpeed - h(receptorsPositions_i, 
                                           receptorsPositions_j, guess)
        
        cost = sum in combinations of ij (loss_ij ** 2)        
    """
    l = array.shape[0]
    m =  1 if np.ndim(guess) == 1 else guess.shape[0] 
    cost = np.zeros((m, 1))
    dists = np.concatenate((np.zeros((1, 1)), tdoa), axis=0) * propSpeed
    for count, c in enumerate( comb(range(l), 2) ):
        (i, j) = (c[0], c[1])
        r1 = array[i]
        r2 = array[j]
        dij = dists[j] - dists[i]
        cost += (dij - h(r1, r2, guess))**2
    return cost

def sph2car(R, phi, theta):
    ''' 
    changes from spherical coordinates to Cartesian coordinates
    R : radius
    phi : azimuth angle in radians
    theta : elevation angle in radians
    
    return ndarray with x,y,z Cartesian coordinates
    '''
    x = R*np.sin(theta)*np.cos(phi)
    y = R*np.sin(theta)*np.sin(phi)
    z = R*np.cos(theta)
    return np.array([x, y, z])

def randParticle(numParticles, dims, radius=1):
    numSensors = dims//3
    r = np.random.rand(numParticles*numSensors) * radius
    phi= 2*np.pi*np.random.rand(numParticles*numSensors)        
    theta = np.pi*np.random.rand(numParticles*numSensors)
    
    return (sph2car(r, phi, theta).T).reshape(numParticles,dims)

def car2sph(x, y, z):
    ''' 
    changes from Cartesian coordinates to spherical coordinates
    obs: theta, phi in degrees
    '''
    if ((x, y, z) == (0, 0, 0)): return np.zeros(3)
    epsilon = 1e-17 # avoid division by zero
    
    R = np.sqrt(x**2 + y**2 + z**2)
    theta_rad = np.arctan(y/(x + epsilon))
    phi_rad = np.arccos(z/(R + epsilon))
    
    theta = (theta_rad/(2*np.pi)) * 360 
    phi = (phi_rad/(2*np.pi)) * 360
    return np.array([R, theta, phi])

def tdoa(source, array, sr=None):
    """
        Calculate the Time Differences of Arrival(TDOA) from the source to the 
        array sensors
        
        source: row 1D or 2D ndarray with the coordinates of the source
        array: row 2D ndarray with the coordinates of all sensors
                
               [xi yi zi] is the coordinates of i-th sensor, i in [1..M]
               1th sensor is the reference
               
               array = [[x1 ... xM]
                        [y1 ... yM]
                        [z1 ... zM]].T
        sr: sampling rate to be simulated(no sampling if sr==None)
        
        return 2D ndarray(shape of (M-1, 1)) with TDOA relative to the 
        reference sensor(array[0])
    """
    toa = dist(source, array)/propSpeed
    tdoa = toa[0] - toa[1:,] 
    if (sr): tdoa = np.round(tdoa*sr)/sr
    return tdoa
