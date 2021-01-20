"""
pso.py

This code is part of Optimization of Hardware Parameters on a Real-Time Sound
Localization System paper

It contains the implementation of Particle Swarm Optimization, the strategy of
finding the best configuration

Authors:
    Matheus Farias
    Davi Moreno
    Daniel Filgueiras
    Edna Barros
"""

import numpy as np
import matplotlib.pyplot as plt
import itertools
import time
import os
import datetime
import mle
from classes import Swarm
import pickle

###############PARAMETERS###################
mics = 4
sampleRate = 56000
particles = 150
maxit = 200
wmax = 0.9
wmin = 0.4
c1 = 2
c2 = 2
### optimization parameters ###
k_mse = 0.6
k_g = 1
N_mse = 4
N_g = 0.49
r_dist = 0.7 #radius parameter
p_dist = 15 * mle.propSpeed/sampleRate #proximity parameter
SWARM_PATH = '' # '' if no swarm to load
###############################
part_dim = mics * 3
ub = [3] * part_dim
lb = [-3] * part_dim
############################################

#Point Cloud
R = np.linspace(1,10,10)#15
phi = np.linspace(0, 2*np.pi, 10, endpoint=False)#24
theta = np.linspace(0, np.pi/2, 5, endpoint=True)#12
nPoints = len(R) * len(phi) * len(theta)

#Cost function
def cost(x):
    mse1 = 0
    mse2 = 0
    array = np.reshape(x,  newshape=(mics, 3))
    #array = np.round(array, decimals=2)
    M = mle.arrayMatrix(array)
    semi_sphere = itertools.product(R, phi, theta)
    for (r, p, t) in semi_sphere:
        sources = mle.sph2car(r, p, t) + np.random.randn(2,3)*0.05
        
        delay1 = mle.tdoa(sources[0], array, sr=sampleRate)
        delay2 = mle.tdoa(sources[1], array, sr=sampleRate)
        
        result1 = mle.mle_hls(delay1, array, M)
        result2 = mle.mle_hls(delay2, array, M)
    
        error1 = float(mle.dist(sources[0], result1))
        error2 = float(mle.dist(sources[1], result2))
        
        mse1 += error1**2
        mse2 += error2**2
    mse = max(mse1, mse2)/nPoints
    
    radius = np.sqrt(np.sum(array**2, axis=1))
    mask = radius > r_dist
    dist_cost = np.sum(((radius-0.7)*mask)**2)
    
    proximity_cost = 0 
    for i,j in itertools.combinations(range(mics), 2):
        d = float(mle.dist(array[i], array[j])) + 1e-17
        proximity_cost += p_dist*(1/d - 1/p_dist) if d < p_dist else 0  
    
    return k_mse*mse/N_mse, k_g*dist_cost/N_g, proximity_cost

now = datetime.datetime.now()
date = now.strftime("%Y-%m-%d-%H:%M:%S")
directory = "simulationsR1/" + f"mic{mics}_sr{sampleRate}_2_" + date

if not os.path.exists(directory):
    os.makedirs(directory)

if SWARM_PATH: print(f"SWARM_PATH: {SWARM_PATH}")
print(f"Number of microphones: {mics}")
print(f"sampleRate: {sampleRate}")
print(f"nPoints: R({len(R)}) * phi({len(phi)}) * theta({len(theta)}) = {nPoints}")
print(f"Iterations: {maxit}")
print(f"Particles: {particles}")
print(f"wmax: {wmax} wmin: {wmin}")
print(f"c1: {c1} c2: {c2}")
print(f"k_mse: {k_mse:.3f} k_g: {k_g:.3f} N_mse: {N_mse:.3f} N_g: {N_g:.3f} r_dist: {r_dist:.3f} p_dist: {p_dist:.3f}")

arquivo = open(directory + "/scores.txt", "w")

if SWARM_PATH: arquivo.write(f"SWARM_PATH: {SWARM_PATH}\n")
arquivo.write(f"Number of microphones: {mics}\n")
arquivo.write(f"sampleRate: {sampleRate}\n")
arquivo.write(f"nPoints: R({len(R)}) * phi({len(phi)}) * theta({len(theta)}) = {nPoints}\n")
arquivo.write(f"Iterations: {maxit}\n")
arquivo.write(f"Particles: {particles}\n")
arquivo.write(f"wmax: {wmax} wmin: {wmin}\n")
arquivo.write(f"c1: {c1} c2: {c2}\n")
arquivo.write(f"k_mse: {k_mse:.3f} k_g: {k_g:.3f} N_mse: {N_mse:.3f} N_g: {N_g:.3f} r_dist: {r_dist:.3f} p_dist: {p_dist:.3f}\n\n")

arquivo.close()

ini = time.time()

ub = np.array(ub)
lb = np.array(lb)
if SWARM_PATH:
    swarm_file = open(SWARM_PATH, 'rb')
    swarm = pickle.load(swarm_file)
    
    positions = swarm.getPositions()
    velocities = swarm.getVelocities()
    best_positions = swarm.getBestPositions()
    best_costs = swarm.getBestCosts()
    current_costs = swarm.getCurrentCosts()
    best_position = swarm.getBestPosition()
    best_cost = swarm.getBestCost()
    best_mse_cost = swarm.getBestMseCost()
    best_dist_cost = swarm.getBestDistCost()
    best_prox_cost = swarm.getBestProxCost()
else:
    positions = mle.randParticle(particles, part_dim, radius=1)
    #positions =np.random.randn(individuos, dimention_individuo) 
    #positions = np.random.rand(particles, part_dim)
    #positions = lb + positions * (ub - lb)
    velocities = np.zeros_like(positions)
    #velocities = np.random.rand(particles, part_dim)
    #velocities = -np.abs(ub - lb) + velocities * 2 * np.abs(ub - lb)
    best_positions = positions.copy()
    best_costs = np.ones(particles) * np.inf
    current_costs = np.zeros(particles)
    best_position = []
    best_cost = np.inf
    best_mse_cost = np.inf
    best_dist_cost = np.inf
    best_prox_cost = np.inf
    for i in range(0,particles):    
        c_mse, c_dist, c_prox = cost(positions[i])
        c = c_mse + c_dist + c_prox
        current_costs[i] = c
        if c < best_costs[i]:
            best_positions[i] = positions[i].copy()
            best_costs[i] = c
            if c < best_cost:
                best_position = positions[i].copy()
                best_cost = c
                best_mse_cost = c_mse
                best_dist_cost = c_dist 
                best_prox_cost = c_prox

global_bests = []
for it in range(0,maxit):    
    if(it==0):
        r0=0
        while(r0== 0 or r0== 0.25 or r0== 0.5 or r0== 0.75):
            r0 = np.random.rand()
            r=r0
    else:
        r = 4*r*(1-r)
        
    w = r*wmin + (wmax-wmin)*it/(maxit)

    r1 = np.random.rand(particles, part_dim)
    r2 = np.random.rand(particles, part_dim)
    
    velocities = w*velocities + c1*r1*(best_positions - positions) + c2*r2*(best_position - positions)
    positions = positions + velocities        
    
    for i in range(0,particles):    
        c_mse, c_dist, c_prox = cost(positions[i])
        c = c_mse + c_dist + c_prox
        current_costs[i] = c
        if c < best_costs[i]:
            best_positions[i] = positions[i].copy()
            best_costs[i] = c
            if c < best_cost:
                best_position = positions[i].copy()
                best_cost = c
                best_mse_cost = c_mse
                best_dist_cost = c_dist 
                best_prox_cost = c_prox
    
    if (it+1) % 5 ==0:
        swarm = Swarm(positions=positions, velocities=velocities, bestPositions=best_positions,
                      bestCosts=best_costs, currentCosts=current_costs, bestPosition=best_position,
                      bestCost=best_cost, bestMseCost=best_mse_cost, bestDistCost=best_dist_cost, 
                      bestProxCost=best_prox_cost)
        now = datetime.datetime.now()
        date = now.strftime(f"%Y-%m-%d-%H:%M:%S")
        f = open(directory + f"/{mics}_{sampleRate}_it{it}_" + date + ".obj", "wb")
        pickle.dump(swarm, f)
        f.close()
    
    max_cost = np.max(best_costs)
    min_cost = np.min(best_costs)
    mean_cost = np.mean(best_costs)
    std_cost = np.std(best_costs)
    
    global_bests.append(best_cost)
    global_best = np.round(best_position, decimals=2).reshape(-1, 3)
    
    s = f"""Iteration {it}, \nbest:{np.array_str(global_best)}, cost:{best_cost:.3f}, mse cost{best_mse_cost:.3f},
    dist cost{best_dist_cost:.3f}, prox cost{best_prox_cost:.3f}\n mean:{mean_cost:.3f}, std:{std_cost:.3f},
    max:{max_cost:.3f}, min:{min_cost:.3f}"""

    arquivo = open(directory + "/scores.txt", "a")
    arquivo.write(s + '\n')
    arquivo.close()
    print(s)

arquivo = open(directory + "/scores.txt", "a")
arquivo.write(f"costs:{global_bests}\n")
arquivo.write("\nBest Geometry:\n")
arquivo.write(np.array_str(global_best)+"\n")
arquivo.write(f"Geometry Cost:{best_cost}\n\n")

end= time.time()
print(f"Time {end-ini} s")
arquivo.write(f"Time {end-ini} s")
arquivo.close()