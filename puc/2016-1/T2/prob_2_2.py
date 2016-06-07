import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
plt.style.use('ggplot')

# Numero de RVs a generar:
N = 10000
# Extraemos v.a. uniformes:
U = st.uniform.rvs(0,1,N)
# Usamos la inversa de la CDF de una normal 
# estandar, su ppf:
X = st.norm.ppf(U)

# Graficamos X:
plt.xlabel('x')
plt.hist(X,bins = 100,normed=True,label = r'Samples using the PIT')
# Graficamos la distribucion normal estandar encima:
x = np.linspace(-5,5,1000)
plt.plot(x,(1./np.sqrt(2.*np.pi))*np.exp(-x**2/2.),'-',label = r'Standard normal PDF')
plt.legend()
plt.show()

