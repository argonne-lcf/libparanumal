import numpy as np
import matplotlib.pyplot as plt

Ndatasets = 20
for n in range(0,Ndatasets+1):
    filename = "advection_%04d.dat"%(n)
    #print(filename)
    x,y,u = np.loadtxt(filename,delimiter=' ',skiprows=1,usecols=(0,1,3),unpack=True)
    indices = np.argsort(x)
    plt.plot(x[indices],u[indices],'b-',label='Lax-Friedrich')

    plt.xlabel('x')
    plt.ylabel('u')
    time = " Time = %d "%(n)
    plt.title(time)
    plt.ylim((-0.2,1.2))
    plt.legend()
    filename = "adv_%04d.png"%(n)
    plt.savefig(filename)
    plt.close()
