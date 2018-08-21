import numpy as np

## This script is for converting the VASP-AIMD result into DeepMD-kit input files.

## coord & force 
D = 4
Atom = 1062
A = Atom + D
f = open("force.txt",'r')
F = f.readlines()
F = np.array(F)

Atom_sub_2 = np.array([])
F_sub_2 = np.array([])
coord = np.array([])
coord = coord.reshape(0,Atom*3)
force = np.array([])
force = force.reshape(0,Atom*3)

for i in range(int(F.shape[0]/A)):
    for j in range(Atom):
        Atom_sub_1 = np.asfarray(np.array([F[D+i*A+j][0:13],F[D+i*A+j][14:27],F[D+i*A+j][13:26]]),float)
        F_sub_1 = np.asfarray(np.array([F[D+i*A+j][39:55],F[D+i*A+j][56:70],F[D+i*A+j][70:84]]),float)
        Atom_sub_2 = np.hstack((Atom_sub_2,Atom_sub_1))
        F_sub_2 = np.hstack((F_sub_2,F_sub_1))
    Atom_sub_2.reshape(1,Atom*3)
    F_sub_2.reshape(1,Atom*3)
    coord = np.vstack((coord,Atom_sub_2))
    force = np.vstack((force,F_sub_2))
    Atom_sub_2 = np.array([])
    F_sub_2 = np.array([])
f.close()        

## box
f = open("POSCAR",'r')
T = f.readlines()
L = float(np.array(T[1]))
X = np.asfarray(np.array([T[2][0:10],T[2][11:21],T[2][22:32]]),float)
Y = np.array(np.array([T[3][0:10],T[3][11:21],T[3][22:32]]),float)
Z = np.array(np.array([T[4][0:10],T[4][11:21],T[4][22:32]]),float)
X = X*L
Y = Y*L
Z = Z*L
XYZ = np.concatenate((X,Y,Z),axis = 0)
XYZ.reshape(1,9)
box = np.array([])
box = box.reshape(0,9)
for i in range(int(F.shape[0]/A)):
    box = np.vstack((box,XYZ))
f.close()
    
    
## energy
f = open("energy.txt",'r')
E = f.readlines()
energy = np.asfarray(np.array(E),float).reshape(int(F.shape[0]/A),1)
f.close()

## file
f1 = open("coord.raw",'w')
np.savetxt(f1,coord,'%2.9f')
f2 = open("box.raw",'w')
np.savetxt(f2,box,'%2.9f')
f3 = open("force.raw",'w')
np.savetxt(f3,force,'%2.9f')
f4 = open("energy.raw",'w')
np.savetxt(f4,energy,'%5.9f')
f1.close()
f2.close()
f3.close()
f4.close()


