"""
classes.py

This code is part of Optimization of Hardware Parameters on a Real-Time Sound
Localization System paper

It contains the definition of Swarm as the main class

Authors:
    Matheus Farias
    Davi Moreno
    Daniel Filgueiras
    Edna Barros
"""

class Swarm:
    def __init__(self, positions, velocities, bestPositions, bestCosts,
                 currentCosts, bestPosition, bestCost, bestMseCost,
                 bestDistCost, bestProxCost):
        self.positions = positions
        self.velocities = velocities
        self.bestPositions = bestPositions
        self.bestCosts = bestCosts
        self.currentCosts = currentCosts
        self.bestPosition = bestPosition
        self.bestCost = bestCost
        self.bestMseCost = bestMseCost
        self.bestDistCost = bestDistCost
        self.bestProxCost = bestProxCost
    
    def setPositions(self, positions): self.positions = positions
    def setVelocities(self, velocities): self.velocities = velocities
    def setBestPositions(self, bestPositions): self.bestPositions = \
                                                    bestPositions
    def setBestCosts(self, bestCosts): self.bestCosts = bestCosts
    def setCurrentCosts(self, currentCosts): self.currentCosts = currentCosts
    def setBestPosition(self, bestPosition): self.bestPosition = bestPosition
    def setBestCost(self, bestCost): self.bestCost = bestCost
    def setBestMseCost(self, bestMseCost): self.bestMseCost = bestMseCost
    def setBestDistCost(self, bestDistCost): self.bestDistCost = bestDistCost
    def setBestProxCost(self, bestProxCost): self.bestProxCost = bestProxCost

    def getPositions(self): return self.positions
    def getVelocities(self): return self.velocities
    def getBestPositions(self): return self.bestPositions
    def getBestCosts(self): return self.bestCosts
    def getCurrentCosts(self): return self.currentCosts
    def getBestPosition(self): return self.bestPosition
    def getBestCost(self): return self.bestCost
    def getBestMseCost(self): return self.bestMseCost
    def getBestDistCost(self): return self.bestDistCost
    def getBestProxCost(self): return self.bestProxCost

