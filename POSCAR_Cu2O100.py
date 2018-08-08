import numpy as np
from numpy import savetxt

a0=np.array([4.31])
i=2
j=2
k=2
Free=np.array(['T','T','T'])
V=np.diag(np.array([i,j,k]))
f=open("POSCAR.txt","w")
f.write("Cu2O100\n")
np.savetxt(f,a0,'%2.9f','\n')
np.savetxt(f,V,'%2.9f',) #V是array不用加'\n'
f.write("Cu O\n")

def POSCAR(i,j,k):
   Cu_origin=np.array([[1/4/i,1/4/j,1/4/k],[3/4/i,3/4/j,1/4/k],[3/4/i,1/4/j,3/4/k],[1/4/i,3/4/j,3/4/k]])
   O_origin=np.array([[0,0,0],[1/2/i,1/2/j,1/2/k]])
   Cu=np.array([])   
   O=np.array([])
   Cu=Cu.reshape(0,3)
   O=O.reshape(0,3)
   for x in range(0,i):
       for y in range(0,j):
           for z in range(0,k):
               Cu_sub=Cu_origin+np.array([[x/i,y/j,z/k],[x/i,y/j,z/k],[x/i,y/j,z/k],[x/i,y/j,z/k]])
               O_sub=O_origin+np.array([[x/i,y/j,z/k],[x/i,y/j,z/k]])
               Cu=np.vstack((Cu,Cu_sub))
               O=np.vstack((O,O_sub))
   return Cu,O


Cu,O=POSCAR(i,j,k)
atom=np.array([Cu.shape[0],O.shape[0]])
atom=atom.reshape(1,2)
np.savetxt(f,atom,'%d')
f.write("Selective dynamics\n")
f.write("Direct\n")
for x in range(0,Cu.shape[0]):
   np.savetxt(f,Cu[x].reshape(1,3),'%2.9f',' ',' ')
   f.write('T T T\n')
for x in range(0,O.shape[0]):
   np.savetxt(f,O[x].reshape(1,3),'%2.9f',' ',' ')
   f.write('T T T\n')
f.close()




