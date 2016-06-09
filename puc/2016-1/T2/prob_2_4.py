import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
plt.style.use('ggplot')

def gen_nrv(N):
    """
    Funcion que genera un vector de N elementos,
    donde cada elemento es extraido de una distribucion
    normal estandar.
    """
    # Extraemos v.a. uniformes:
    U = st.uniform.rvs(0,1,N)
    # Usamos la inversa de la CDF de una normal 
    # estandar, su ppf:
    X = st.norm.ppf(U)
    return X

def sq_exp_kernel(x,N,epsilon = 1e-6):
    """
    Funcion que retorna una matriz de NxN, donde el elemento 
    (i,j) es obtenido de:

              exp(-0.5*(x[i]-x[j])**2)

    El termino epsilon = 1e-6 se agrega a la diagonal para asegurar
    estabilidad numerica.
    """
    Sigma = np.zeros([N,N])
    for i in range(N):
        for j in range(N):
            if i != j:
                Sigma[i,j] = np.exp(-0.5*((x[i]-x[j])**2))
            else:
                Sigma[i,i] = 1 + epsilon
    return Sigma
 
def gen_mnrv(mu,Sigma,N):
    """
    Algoritmo que genera un vector aleatorio de dimension N, donde 
    el mismo es extraido de una distribucion normal multivariada con 
    media mu y matriz de covarianza Sigma.
    """
    # Encontramos la descomposicion de Cholesky 
    # de la matriz de covarianza:
    L = np.linalg.cholesky(Sigma)

    # Generamos el vector aleatorio S:
    S = gen_nrv(N)
    # Generamos el vector X:
    return mu + np.dot(L,S)

N = 100
# Obtenemos la matriz de covarianza:
x = np.linspace(-5,5,N)
Sigma = sq_exp_kernel(x,N)
# Generamos varios vectores aleatorios 
# extraidos de dist. gaussiana 
# multivariada usando nuestro algoritmo y 
# el de Numpy:
Nsim = 5000
x1 = np.zeros(Nsim)
x2 = np.zeros(Nsim)
x1_numpy = np.zeros(Nsim)
x2_numpy = np.zeros(Nsim)
mu = np.ones(N)*5.

Xsim = np.zeros([Nsim,N])
for i in range(Nsim):
    X = gen_mnrv(mu,Sigma,N)
    Xsim[i,:] = X
    Xnumpy = np.random.multivariate_normal(mu,Sigma)
    x1[i] = X[1]
    x2[i] = X[2]
    x1_numpy[i] = Xnumpy[1]
    x2_numpy[i] = Xnumpy[2]

plt.subplot(222)
plt.plot(x1,x2,'.',alpha=0.1)
plt.plot(x1_numpy,x2_numpy,'.',alpha=0.1)
plt.legend()
plt.subplot(224)
plt.hist(x1,bins=50,label='Nuestro algoritmo',alpha=0.5,normed=True)
plt.hist(x1_numpy,alpha=0.5,bins=50,label='Algoritmo de Numpy',normed=True)
plt.xlabel(r'$\vec{X}_2$')
plt.legend()
plt.subplot(221)
plt.ylabel(r'$\vec{X}_3$')
plt.hist(x2,bins=50,alpha=0.5,normed=True,orientation=u'horizontal')
plt.hist(x2_numpy,alpha=0.5,bins=50,normed=True,orientation=u'horizontal')
plt.show()

for i in range(30):
    plt.plot(x,Xsim[i,:])
plt.xlabel(r'$\vec{x}$')
plt.ylabel(r'$\vec{X}$')
plt.show()
