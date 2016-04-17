import numpy as np

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

print 'Probabilidad de observar X=18:',\
      str(np.round(binom_dist(18.,33.,0.5)*100,2))+'%'
