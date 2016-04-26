# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import utils
import numpy as np

def PDF(X,N,r):
    # Para cada valor de r, evaluamos p(r|X), donde 
    # lo mas complejo es evaluar el denominador. El numerador es simple; 
    # es evaluar la binomial que vimos en el problema 2b para todos 
    # los valores de r y multiplicar eso por uno:
    numerador = utils.binom_dist(X,N,r)

    # Para el denominador, notamos que es simplemente un numero. Para 
    # calcularlo, ocupamos la funcion trapz de numpy, a la que se le 
    # pasa el integrando evaluado en todo el rango deseado, el rango 
    # donde estan evaluados los puntos y listo. Asi, integramos de 0 a 1:
    rint = np.linspace(0,1,1000)
    denominador = np.trapz(utils.binom_dist(X,N,rint),x=rint)
    
    return numerador/denominador

def CDF(X,N,r):
    # Integramos la PDF desde 0 hasta cada valor de r:
    cdf = np.zeros(len(r))
    for i in range(len(r)):
        c_r = np.linspace(0,r[i],100)
        pdf = PDF(X,N,c_r)
        cdf[i] = np.trapz(pdf,x=c_r)
    return cdf

# Primero, definimos el valor observado del numero de exitos e intentos...
X = 18.
N = 33.

# Definimos los valores de r a integrar: desde 0 a 0.5:
r = np.linspace(0.0,0.5,1000)

# Calculamos la integral de la PDF desde 0 a 0.5:
pdf = PDF(X,N,r)

p1 = np.trapz(pdf,r)

# Repetimos el calculo para la integral desde 0.5 a 1:
r = np.linspace(0.5,1,1000)
pdf = PDF(X,N,r)
p2 = np.trapz(pdf,r)

# Comparamos los valores:
print '\n\t Resultado problema 2, parte d):'
print '\t  p(r>0.5|X) = ',p2
print '\t  p(r<0.5|X) = ',p1
print '\n\t Razon entre ambos numeros: '
print '\t p(r>0.5|X)/p(r<0.5|X) = ',p2/p1
print '\n'
