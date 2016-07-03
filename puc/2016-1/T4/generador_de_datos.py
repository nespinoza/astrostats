import numpy as np

N = 300
noise_level = 30 # in ppm
# Generate times:
t = np.random.uniform(0.2,0.8,N)

# Generate baseline model:
A1 = 0.3
B1 = -0.1
A2 = 0.1
B2 = -0.3
P = 0.7
P2 = 1.2
t = t[np.argsort(t)]
polynomial = (A1*np.sin(t*(2.*np.pi/P)) + B1*np.cos(t*(2.*np.pi/P)) +\
             A2*np.sin(t*(2.*np.pi/P2)) + B2*np.cos(t*(2.*np.pi/P2)))*3e-4

polynomial = polynomial - np.median(polynomial)

# Generate transit:
transit = np.ones(len(t))
idx = np.where((t>0.4)&(t<0.7))[0]
transit[idx] = np.ones(len(idx))*0.9999

# Generate noise:
noise = np.random.normal(0,noise_level,len(t))

from pylab import *
model = polynomial + transit
data = model + noise*1e-6
plot(t,data,'.')
plot(t,model)
show()

f = open('data.dat','w')
f.write('# t - 2457518  flujo         error en el flujo\n')
for i in range(len(t)):
    f.write(str(t[i])+' '+str(data[i])+' '+str(noise_level*1e-6)+'\n')
f.close()
f = open('model.dat','w')
f.write('# t - 2457518  flujo\n')
for i in range(len(t)):
    f.write(str(t[i])+' '+str(model[i])+'\n')
f.close()
