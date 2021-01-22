"""
cloud_matches.py

This code is part of Optimization of Hardware Parameters on a Real-Time Sound
Localization System paper.

It contains the implementation of the comparison of two cloud of points.

Authors:
    Matheus Farias
    Davi Moreno
    Daniel Filgueiras
    Edna Barros
"""

import numpy as np
import numpy.linalg as LA
import scipy.optimize as OPT
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from bestarraysR2 import *

#3D rotations matrix
#line matrix
def rotX(theta):
    Rx=np.array([[1,0,0],
                 [0,np.cos(theta),np.sin(theta)],
                 [0,-np.sin(theta),np.cos(theta)]])

    return Rx

#line matrix
def rotY(theta):
    Ry=np.array([[np.cos(theta),0,-np.sin(theta)],
                  [0,1,0,],
                  [np.sin(theta),0,np.cos(theta)]])

    return Ry

#line matrix
def rotZ(theta):
    Rz=np.array([[np.cos(theta),np.sin(theta),0],
                  [-np.sin(theta),np.cos(theta),0],
                  [0,0,1]])
    return Rz


#routine to clean matrix multiplication
def clearM(X):
    X=np.asarray(X)
    [M,N]=X.shape
    for i in range(M):
        for j in range(N):
            if abs(X[i,j])<1.0e-15:
                X[i,j]=0.0
            
  
         
#line vectors
#do not verify vector size
#utilize first vector dimensions and assume the second share the same dimension
def dist2(x0,x1):
    N=x0.size
    s=0
    for i in range(N):
        d=x0[i]-x1[i]
        s+=d*d
    
    return s

#remove the average of a set of line vectors
def centerM(X):
    X0=np.copy(X)
    [M,N]=X.shape
    xm=np.mean(X,0)
    
    for i in range(M):
        X0[i,:]-=xm
        
    return X0,xm


#error metric, a modification of standard SSE
#the result is the sum of distance of the closes points to X1 and X2
#line vector
def SSE(X1,X2):
    [M1,N1]=X1.shape
    [M2,N2]=X2.shape
    if N1!=N2:
        print("vectors do not share the same dimension!!!!")
    if M1!=M2:
        print("the set of points do not share the same cardinality!!!")
        
    idx_v=np.zeros(M2)
    s=0
    for i in range(M1):
        for j in range(M2):
            if idx_v[j]==0:
                dmin=dist2(X1[i,:],X2[j,:])
                break
        d=0
        for j in range(M2):
            if idx_v[j]==0:
                d=dist2(X1[i,:],X2[j,:])
                if d<dmin:
                    dmin=d
                    idx_v[j]=1
        s+=dmin
        
    return s

#test set
#square
Q1=np.array([[0,0,0],[2.0,0,0],[2.0,2.0,0],[0.0,2.0,0],[0,0,0]])
centerM(Q1)
Q2=Q1+[1,1,1]
Q3=Q1@rotZ(-np.pi*78.0/180.0)
Q4=1.5*Q1@rotX(-np.pi*120/180.0)@rotY(-np.pi*40/180.0)@rotZ(-np.pi*0/180.0)+[1,1,1]

#house
C1=np.array([[0,0,0],[2.0,0,0],[2.0,2.0,0],[1.0,3.0,0],[0,2.0,0],[0,0,0]])
centerM(C1)
C2=C1+[1,1,1]
C3=C1@rotZ(-np.pi*78.0/180.0)
C4=1.5*C1@rotX(-np.pi*120/180.0)@rotY(-np.pi*40/180.0)@rotZ(-np.pi*0/180.0)+[1,1,1]

#pyramid
P1=np.array([[0,0,0],[2.0,0,0],[2.0,2.0,0],[0.0,2.0,0],[0,0,0],[1.,1.,1.],[2.,2.,0.]])
centerM(P1)
P2=P1+[1,1,1]
P3=P1@rotZ(-np.pi*78.0/180.0)
P4=2.5*P1@rotX(-np.pi*60/180.0)@rotY(-np.pi*60/180.0)@rotZ(-np.pi*60/180.0)+[1,2,1.5]


#lines
R1=np.array([[2,0,0],[1.5,0,0],[1,0,0]])
R2=np.array([[0,2,0],[0,1.5,0],[0,1,0]])

min_mic8_sr40k = np.array([[ 0.02, -0.03,  0.02],
       [ 0.35, -0.29,  0.36],
       [ 0.64, -0.06, -0.25],
       [-0.66,  0.14,  0.2 ],
       [ 0.12,  0.34, -0.6 ],
       [-0.26, -0.18,  0.52],
       [ 0.01, -0.18,  0.33],
       [ 0.54, -0.27,  0.16]])

min_R1_mic4_sr40k = np.array([[ 0.11,  0.17, -0.13],
                           [-0.55, -0.34, -0.11],
                           [ 0.71, -0.06, -0.09],
                           [-0.13,  0.67, -0.07]])
min_R1_mic6_sr40k = np.array([[-0.14, -0.01,  0.11],
                           [-0.33,  0.61,  0.14],
                           [-0.27, -0.36,  0.58],
                           [ 0.05, -0.07, -0.11],
                           [ 0.36, -0.57, -0.24],
                           [ 0.31,  0.18, -0.62]])
min_R1_mic8_sr40k = np.array([[ 0.02, -0.03,  0.02],
                           [ 0.35, -0.29,  0.36],
                           [ 0.64, -0.06, -0.25],
                           [-0.66,  0.14,  0.2 ],
                           [ 0.12,  0.34, -0.6 ],
                           [-0.26, -0.18,  0.52],
                           [ 0.01, -0.18,  0.33],
                           [ 0.54, -0.27,  0.16]])# SSE_min 0.4035620234833867
    
min_R2_mic4_sr32k = np.array([[-0.11, -0.02, -0.01],
                           [ 0.16,  0.24, -0.67],
                           [-0.08, -0.63, -0.29],
                           [ 0.01,  0.08,  0.71]])#SSE_min 0.05275168275119908
min_R2_mic6_sr32k = np.array([[-0.11, -0.03, -0.01],
                           [-0.05,  0.54,  0.42],
                           [-0.08,  0.06,  0.7 ],
                           [ 0.02, -0.68,  0.03],
                           [ 0.12, -0.56, -0.41],
                           [ 0.17,  0.25, -0.61]])# SSE_min 0.11330461376649809
min_R2_mic8_sr32k = np.array([[ 0.09,  0.01,  0.08],
                           [ 0.22, -0.6 , -0.31],
                           [-0.15, -0.01,  0.69],
                           [-0.08, -0.27,  0.62],
                           [ 0.06,  0.48, -0.5 ],
                           [-0.17,  0.34,  0.58],
                           [-0.14,  0.58,  0.28],
                           [ 0.16,  0.13, -0.67]])# SSE_min 0.13819021199110468

#substitute here what you want to compare, X1 is the reference set
X1=mic8_sr24k#mic8_sr32k
X2=min_R2_mic8_sr32k#min_R2_mic6_sr32k min_mic8_sr40k
filename = "R2_match_geometry_8"
                
X10,xm1=centerM(X1)
X20,xm2=centerM(X2)

pi2=np.pi*2

def matchingPoints(v):
    
    Rx=rotX(v[0]%pi2)
    Ry=rotY(v[1]%pi2)
    Rz=rotZ(v[2]%pi2)
    X22=np.copy(X20)
    X22=(X22@Rx@Ry@Rz)*v[3]
    clearM(X22)
    r=SSE(X10,X22)
    
    return r


#generate a list of initial points to do the optimization
vl=list()
N=40
Astep=2*np.pi/N
for ix in range(N):
    ax=ix*Astep
    for iy in range(N):
        ay=iy*Astep
        for iz in range(N):
            az=iz*Astep
            vl.append([ax,ay,az,1])

#select the best initial point for the optimization
vl=np.asarray(vl)
[M,N]=vl.shape
idxMin=0
dMin=matchingPoints(vl[0])
for idx in range(1,M):
    d=matchingPoints(vl[idx])
    #print("idx=",idx," d=",d)
    if d<dMin:
        dMin=d
        idxMin=idx

#do the optimization
results=OPT.minimize(matchingPoints,vl[idxMin],method="Nelder-Mead",options={"disp":True,"maxiter": 2000,"xatol":1e-15,"fatol":1e-15})

#apply the optimization transforms in a copy of X2
#if the optimization went successful, and X1 and X2 were rotations and scalings
#one another, Xc may have similar values to X1
Xc=np.copy(X20)
Rx=rotX(results.x[0]%pi2)
Ry=rotY(results.x[1]%pi2)
Rz=rotZ(results.x[2]%pi2)
Xc=(Xc@Rx@Ry@Rz)*results.x[3]
clearM(Xc)

#centering Xc to X1
Xc+=xm1

#plotting the results in X-Y
fig = plt.figure()

ax=fig.add_subplot(221, projection='3d') 

ax.plot3D(X1[:,0],X1[:,1],X1[:,2],'d-',label="X1" ) 
ax.plot3D(X2[:,0],X2[:,1],X2[:,2],'o-',label="X2" ) 
ax.plot3D(Xc[:,0],Xc[:,1],Xc[:,2],'+-',label="Xc" )
plt.subplot(222)
plt.plot(X1[:,0],X1[:,1],'d-',label="X1" ) 
plt.plot(X2[:,0],X2[:,1],'o-',label="X2" ) 
plt.plot(Xc[:,0],Xc[:,1],'+-',label="Xc" )
plt.xlabel("x")
plt.ylabel("y")
plt.subplot(223)
plt.plot(X1[:,1],X1[:,2],'d-',label="X1" ) 
plt.plot(X2[:,1],X2[:,2],'o-',label="X2" ) 
plt.plot(Xc[:,1],Xc[:,2],'+-',label="Xc" )
plt.xlabel("y")
plt.ylabel("z")
plt.subplot(224)
plt.plot(X1[:,0],X1[:,2],'d-',label="X1" ) 
plt.plot(X2[:,0],X2[:,2],'o-',label="X2" ) 
plt.plot(Xc[:,0],Xc[:,2],'+-',label="Xc" )
plt.xlabel("x")
plt.ylabel("z")
plt.xlim(-3, 3)
plt.ylim(-3, 3)
ax.axis('equal')
plt.legend()
plt.show()

#print the results
rx=results.x[0]*180.0/np.pi
ry=results.x[1]*180.0/np.pi
rz=results.x[2]*180.0/np.pi
s=results.x[3]

print("Rotation in x:",rx%360)
print("Rotation in y:",ry%360)
print("Rotation in z:",rz%360)
print("Scaling:",s)
print("SSE=",results.fun)

############################################################################
l_width=1
array=X1
array2=Xc#(Xc-xm1)@rotY(pi2/2)@rotX(-pi2/4) + xm1
plt.rcParams.update({'font.size': 18, "text.usetex": True, 'text.latex.preamble' : [r'\usepackage{amsmath}',r'\usepackage{amssymb}'] })
fig = plt.figure(1, figsize=(9, 6))
ax = plt.axes(projection='3d')
ax.view_init(30, 45)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
#ax.set_xlim([-1.25, 1.25])
#ax.set_ylim([-1.25, 1.25])
#ax.set_zlim([-1.25, 1.25])
#for i in range(array.shape[0]):
#    ax.text(array[i,0], array[i,1], array[i,2], f"{i}", zorder=1)
if(len(X1)==4):
    ax.scatter3D(array[:,0], array[:,1], array[:,2], c='darkblue', depthshade=False , s=40)
    ax.plot([array[0,0], array[1,0]], [array[0,1], array[1,1]], [array[0,2], array[1,2]], c="blue", linewidth=l_width)
    ax.plot([array[0,0], array[2,0]], [array[0,1], array[2,1]], [array[0,2], array[2,2]], c="blue", linewidth=l_width)
    ax.plot([array[0,0], array[3,0]], [array[0,1], array[3,1]], [array[0,2], array[3,2]], c="blue", linewidth=l_width)

    ax.scatter3D(array2[:,0], array2[:,1], array2[:,2], c='darkgreen', depthshade=False , s=40)
    ax.plot([array2[0,0], array2[1,0]], [array2[0,1], array2[1,1]], [array2[0,2], array2[1,2]], c="green", linewidth=l_width)
    ax.plot([array2[0,0], array2[2,0]], [array2[0,1], array2[2,1]], [array2[0,2], array2[2,2]], c="green", linewidth=l_width)
    ax.plot([array2[0,0], array2[3,0]], [array2[0,1], array2[3,1]], [array2[0,2], array2[3,2]], c="green", linewidth=l_width)
elif(len(X1)==6):
    ax.scatter3D(array[:,0], array[:,1], array[:,2], c='darkblue', depthshade=False , s=40)
    ax.plot([array[0,0], array[1,0]], [array[0,1], array[1,1]], [array[0,2], array[1,2]], c="blue", linewidth=l_width)
    ax.plot([array[0,0], array[2,0]], [array[0,1], array[2,1]], [array[0,2], array[2,2]], c="blue", linewidth=l_width)
    ax.plot([array[0,0], array[3,0]], [array[0,1], array[3,1]], [array[0,2], array[3,2]], c="blue", linewidth=l_width)
    ax.plot([array[0,0], array[4,0]], [array[0,1], array[4,1]], [array[0,2], array[4,2]], c="blue", linewidth=l_width)
    ax.plot([array[0,0], array[5,0]], [array[0,1], array[5,1]], [array[0,2], array[5,2]], c="blue", linewidth=l_width)

    ax.scatter3D(array2[:,0], array2[:,1], array2[:,2], c='darkgreen', depthshade=False , s=40)
    ax.plot([array2[0,0], array2[1,0]], [array2[0,1], array2[1,1]], [array2[0,2], array2[1,2]], c="green", linewidth=l_width)
    ax.plot([array2[0,0], array2[2,0]], [array2[0,1], array2[2,1]], [array2[0,2], array2[2,2]], c="green", linewidth=l_width)
    ax.plot([array2[0,0], array2[3,0]], [array2[0,1], array2[3,1]], [array2[0,2], array2[3,2]], c="green", linewidth=l_width)
    ax.plot([array2[0,0], array2[4,0]], [array2[0,1], array2[4,1]], [array2[0,2], array2[4,2]], c="green", linewidth=l_width)
    ax.plot([array2[0,0], array2[5,0]], [array2[0,1], array2[5,1]], [array2[0,2], array2[5,2]], c="green", linewidth=l_width)
elif(len(X1)==8):
    ax.scatter3D(array[:,0], array[:,1], array[:,2], c='darkblue', depthshade=False , s=40)
    ax.plot([array[0,0], array[1,0]], [array[0,1], array[1,1]], [array[0,2], array[1,2]], c="blue", linewidth=l_width)
    ax.plot([array[0,0], array[2,0]], [array[0,1], array[2,1]], [array[0,2], array[2,2]], c="blue", linewidth=l_width)
    ax.plot([array[0,0], array[3,0]], [array[0,1], array[3,1]], [array[0,2], array[3,2]], c="blue", linewidth=l_width)
    ax.plot([array[0,0], array[4,0]], [array[0,1], array[4,1]], [array[0,2], array[4,2]], c="blue", linewidth=l_width)
    ax.plot([array[0,0], array[5,0]], [array[0,1], array[5,1]], [array[0,2], array[5,2]], c="blue", linewidth=l_width)
    ax.plot([array[0,0], array[6,0]], [array[0,1], array[6,1]], [array[0,2], array[6,2]], c="blue", linewidth=l_width)
    ax.plot([array[0,0], array[7,0]], [array[0,1], array[7,1]], [array[0,2], array[7,2]], c="blue", linewidth=l_width)

    ax.scatter3D(array2[:,0], array2[:,1], array2[:,2], c='darkgreen', depthshade=False , s=40)
    ax.plot([array2[0,0], array2[1,0]], [array2[0,1], array2[1,1]], [array2[0,2], array2[1,2]], c="green", linewidth=l_width)
    ax.plot([array2[0,0], array2[2,0]], [array2[0,1], array2[2,1]], [array2[0,2], array2[2,2]], c="green", linewidth=l_width)
    ax.plot([array2[0,0], array2[3,0]], [array2[0,1], array2[3,1]], [array2[0,2], array2[3,2]], c="green", linewidth=l_width)
    ax.plot([array2[0,0], array2[4,0]], [array2[0,1], array2[4,1]], [array2[0,2], array2[4,2]], c="green", linewidth=l_width)
    ax.plot([array2[0,0], array2[5,0]], [array2[0,1], array2[5,1]], [array2[0,2], array2[5,2]], c="green", linewidth=l_width)
    ax.plot([array2[0,0], array2[6,0]], [array2[0,1], array2[6,1]], [array2[0,2], array2[6,2]], c="green", linewidth=l_width)
    ax.plot([array2[0,0], array2[7,0]], [array2[0,1], array2[7,1]], [array2[0,2], array2[7,2]], c="green", linewidth=l_width)

ax.set_xlim([-1.25, 1.25])
ax.set_ylim([-1.25, 1.25])
ax.set_zlim([-1.25, 1.25])

#ax.set_xticks(xy_major_ticks)
#ax.set_yticks(xy_major_ticks)
#ax.set_zticks(z_major_ticks)
plt.tight_layout()
plt.savefig(f"{filename}.eps", format="eps", dpi=1000)
plt.show()
