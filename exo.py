"""/*

//////////////////////////////////

EXO 5 :
Dans cet exercice et les suivants on va réimplementer une régression linéaire depuis zéro. 

Dans cet exercice on va coder la fonction qui calcule la sortie du modèle. 

Coder la fonction predict_linear(x, theta0, theta1) pour une régression linéaire utilisant une variable : 
y
^
=
f
θ
(
x
)
=
θ
0
+
θ
1
∗
x
y
^
​
 =f 
θ
​
 (x)=θ 
0
​
 +θ 
1
​
 ∗x

Remarque : dans cet exercice et les suivant on utilisera numpy à chaque fois qu'on manipulera des vecteurs ou des matrices. 

par exemple, ici on considère que x est un vecteur numpy




en python"""


import numpy as np


def predict_linear(x, theta0, theta1):

     
    return theta0 + theta1 * x


def grad(x, y, theta0, theta1):
    """
    Calcule le gradient de la MSE pour une régression linéaire simple.

    On utilise la MSE définie comme MSE = (1/m) * sum((y_hat - y)**2)
    où y_hat = theta0 + theta1 * x.

    Le gradient renvoyé est un vecteur numpy [dtheta0, dtheta1].
    """
    x = np.asarray(x)
    y = np.asarray(y)
    m = x.size
    if m == 0:
        raise ValueError("x and y must contain at least one element")
    y_hat = theta0 + theta1 * x
    error = y_hat - y
    dtheta0 = (2.0 / m) * np.sum(error)
    dtheta1 = (2.0 / m) * np.sum(error * x)
    return np.array([dtheta0, dtheta1])







""""Pour trouver les paramètres d'un modèle, il est habituel d'utiliser la Mean Squared Error (MSE) comme fonction de coût. 

en partant du model precedent

Coder fonction mse(predictions, labels) qui calcule la MSE entre les prédictions d'un modèle et les labels

 

predictions et labels seront des vecteurs numpy"""

def mse(predictions, labels):
    return np.mean((predictions - labels) ** 2)


"""Coder la fonction grad(x, y, theta0 ,theta1)

qui calcule le gradient de la fonction de coût pour la régression linéaire (la MSE). Le type de retour devra être un vecteur numpy : le premier élement contiendra la dérivée par rapport à 
θ
0
θ 
0
​
  et le second élément contiendra la dérivée par rapport à 
θ
1
θ 
1
​
 

 

Vérifier bien que vous renvoyez bien un vecteur simple. 

On prendra pour la formule de la MSE  
M
S
E
=
1
/
m
∑
(
…
)
MSE=1/m∑(…) 

x, y seront des tableaux numpy

theta0, theta1 juste des nombres"""

import numpy as np

def grad(x, y, theta0, theta1):
   
    m = x.size
    
    y_hat = theta0 + theta1 * x
    error = y_hat - y
    dtheta0 = (2.0 / m) * np.sum(error)
    dtheta1 = (2.0 / m) * np.sum(error * x)
    return np.array([dtheta0, dtheta1])

"""coder la fonction compute_theta_sgd(theta0, theta1, learning_rate,x ,y) : np.array

et qui renvoie les nouveaux theta après une seule itération de la descente de gradient (pas de boucle for). 

 

 

La fonction grad de l'exercice précédent est disponible"""

def compute_theta_sgd(theta0, theta1, learning_rate, x, y):
    gradients = grad(x, y, theta0, theta1)
    new_theta0 = theta0 - learning_rate * gradients[0]
    new_theta1 = theta1 - learning_rate * gradients[1]
    return np.array([new_theta0, new_theta1])





"""6. Régression linéaire à une variable V - entraînement
Reprendre les exercices précédent et faire une fonction gradient_descent(X, y, theta0, theta1, learning_rate, n_iterations) 

pour effectuer la descente de gradient et qui retourne theta0 et theta1 après l'entraînement (sous forme de vecteur 1D numpy)"""

def gradient_descent(X, y, theta0, theta1, learning_rate, n_iterations):
    for _ in range(n_iterations):
        gradients = grad(X, y, theta0, theta1)
        theta0 -= learning_rate * gradients[0]
        theta1 -= learning_rate * gradients[1]
    return np.array([theta0, theta1])
