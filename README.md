# Optimization of Hardware Parameters on a Real-Time Sound Localization System

<p align="center">
  <img height = "300" width = auto src="https://raw.githubusercontent.com/matheussfarias/optimizationssl/master/arrayexample.jpeg">
</p>

Sound source localization (SSL) systems have applications in several areas, including military, home security, robotics, and biology. For most of the systems, the time difference of arrival (TDOA) of the signal between the sensors of a geometry of well-defined sensors is used to perform localization. From a theoretical view, this method becomes the problem of solving a set of hyperbolic equations. In practice, these equations are related to several hardware features, such as **1**) the geometry of the sensors' disposition, **2**) the total number of sensors, and **3**) the sampling rate of the received signals. There is also a lack of study on the relevance of the above factors on determining novel well-suitable designs. Therefore, this paper proposes a mathematical model for evaluating SSL systems' arrays and presents a methodology for optimizing the proposed model's hardware parameters. The model is built considering trade-offs between accurate predictions and the amount of hardware used to achieve it. The optimization procedure is done by the Particle Swarm Optimization (PSO) algorithm, which does not require cost function's gradient and is used in similar approaches. We applied this approach in both indoor and outdoor environments. Our results showed competitive improvements after comparing to other traditional geometries found in related works, and performed up to 33.0% better than similar problem approaches.

## Code

This repository contains the code used in the paper. 

**pso** contains the implementation of Particle Swarm Optimization, used to as strategy to find the best configuration.

**classes** and **mle** are imports with functions/classes that are used in the main code.

**cloud_matches** contains the matching evaluation strategy between two arrays.

## Authors

The paper was written by:
* **Matheus Farias**
* **Davi Moreno**
* **Daniel Filgueiras**
* **Edna Barros**