# Optimization of Hardware Parameters on a Real-Time Sound Localization System

<p align="center">
  <img height = "300" width = auto src="https://raw.githubusercontent.com/matheussfarias/optimizationssl/master/arrayexample.jpeg">
</p>

Sound source localization (SSL) systems have applications in several areas, including military, home security, robotics, and biology. For such systems, the time difference of arrival (TDOA) of the signal between the sensors of the geometry of well-defined sensors is used to perform localization. From a theoretical view, this method becomes the problem of solving a set of hyperbolic equations. In practice, these equations are related to several factors associated with the hardware of the system, such as **1**) the geometry of the sensors' disposition, **2**) the total number of sensors, and **3**) the sampling rate of the received signals. However, there is not an analysis of the relationship between environmental conditions and geometrical configurations.  There is also no study on the relevance of the above factors on determining novel well-suitable designs. Therefore, this paper proposes a mathematical model for evaluating SSL systems' arrays and presents a methodology for optimizing the proposed model's hardware parameters. The model is built considering trade-offs between accurate predictions and the amount of hardware used to achieve it. The optimization procedure is done by the Particle Swarm Optimization (PSO) algorithm, which does not require previous knowledge of the cost function and is used in similar approaches. We applied this approach in both indoor and outdoor environments. After comparing other traditional geometries found in related works, and other similar problem approaches, our results showed competitive improvements.

## Code

This repository contains the code used in the paper. 

**pso.py** contains the implementation of Particle Swarm Optimization, used to as strategy to find the best configuration

**classes.py** and **mle.py** are imports with functions/classes that are used in the main code.

## Authors

The paper was written by:
* **Matheus Farias**
* **Davi Moreno**
* **Daniel Filgueiras**
* **Edna Barros**