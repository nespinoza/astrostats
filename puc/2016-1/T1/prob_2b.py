# -*- coding: utf-8 -*-
import numpy as np
import random

# Esta funcion calcula un termino binomial de N 
# sobre k:
def binomial_term(N,k):
    f1 = np.sum(np.log(np.arange(N,0,-1)))
    f2 = np.sum(np.log(np.arange(N-k,0,-1)))
    f3 = np.sum(np.log(np.arange(k,0,-1)))
    return np.exp(f1-f2-f3)

# Esta funcion retorna la probabilidad de tener x 
# exitos en N eventos, donde la probabildiad de 
# exito es r:
def binom_dist(x,N,r):
    return binomial_term(N,x)*(r**x)*(r**(N-x))

r = 0.5
N = 33
x = 18
nsim = 1000

print '\n\t     Codigo para el P2, parte b \n'
P = binom_dist(x,N,r)*100
print '\t > Probabilidad de observar X=18:',\
      str(np.round(P,1))+'%'

print '\t > Corriendo simulacion asumiendo r = 0.5...'
n18 = 0.0
for i in range(nsim):
    exitos = 0.0
    for j in range(N):
        exitos += random.randint(0,1)
    if exitos == 18.:
        n18 += 1.
print '\t    La probabilidad simulada es ',\
          str(np.round(n18/np.double(nsim)*100,1))+'%\n'

