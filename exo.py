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





def mse(predictions, labels):
    return np.mean((predictions - labels) ** 2)

def grad(x, y, theta0, theta1):
   
    m = x.size
    
    y_hat = theta0 + theta1 * x
    error = y_hat - y
    dtheta0 = (2.0 / m) * np.sum(error)
    dtheta1 = (2.0 / m) * np.sum(error * x)
    return np.array([dtheta0, dtheta1])



 

 


def compute_theta_sgd(theta0, theta1, learning_rate, x, y):
    gradients = grad(x, y, theta0, theta1)
    new_theta0 = theta0 - learning_rate * gradients[0]
    new_theta1 = theta1 - learning_rate * gradients[1]
    return np.array([new_theta0, new_theta1])



"""6. Régression linéaire à une variable V - entraînement
Reprendre les exercices précédent et faire une fonction gradient_descent(X, y, theta0, theta1, learning_rate, n_iterations) 

pour effectuer la descente de gradient et qui retourne theta0 et theta1 après l'entraînement (sous forme de vecteur 1D numpy)"""

def gradient_descent(x, y, theta0, theta1, learning_rate, n_iterations):
    for _ in range(n_iterations):
        theta = compute_theta_sgd(theta0, theta1, learning_rate, x, y)
        theta0, theta1 = theta[0], theta[1]
    return np.array([theta0, theta1])




"""Calcul dérivée partielle et gradient 3
On considère la fonction suivante $ f(x, y, z) = - (x - y)^2 + e^{yz} $


2. Coder la fonction f_gradient(x: np.array, y:np.array, z:np.array) qui retourne le gradient de f
sous forme de tableau numpy 1D (vecteur numpy)"""

def f_gradient(x: np.array, y: np.array, z: np.array):
    df_dx = -2 * (x - y)
    df_dy = 2 * (x - y) + z * np.exp(y * z)
    df_dz = y * np.exp(y * z)
    return np.array([df_dx, df_dy, df_dz])

# test
x = np.array([1.0, 2.0, 3.0])
y = np.array([0.0, 1.0, 2.0])
z = np.array([1.0, 1.0, 1.0])
gradient = f_gradient(x, y, z)
print(gradient)  # Affiche le gradient de f pour les valeurs données de x, y et z


"""Numpy - création vecteur et matrice 2
1. Créer un vecteur numpy avec les valeurs [33, 10, 200] et la stocker dans la variable vector1
2. Créer un vecteur numpy avec les valeur [-1, 0, 1] et la stocker dans la variable vector2
3. Calculer la soustraction entre vector1 et vector2 et stocker le résultat dans vector3
4. Créer une matrice numpy avec les valeurs [[2, 2000], [1000, -100]] et la stocker dans la variable
matrix"""

vector1 = np.array([33, 10, 200])
vector2 = np.array([-1, 0, 1])
vector3 = vector1 - vector2
matrix = np.array([[2, 2000], [1000, -100]])
print("vector3:", vector3)
print("matrix:\n", matrix)

"""Predictive maintenance III : Feature engineering
Linear models rarely work without feature engineering. Feature engineering is the art of
transforming one's data to make the model perform better.
1. Create a df variable containing the "predictive_maintenance.csv" file as a pandas dataframe.
2. Do a train_test_split of the df dataframe and store the result in the train and test
3. Create a variable x_train containing all columns except the variable to be predicted. Create a
y_train variable containing the variable to be predicted.
4. Using the MinMaxScaler class, normalize the numeric columns of the x_train variable (vibration,
age, avg_speed, last_revision. Store the result in a variable called X_normalized.
6. Create a model variable that is an instance of the LogisticRegression 
7. Train the model on the normalized data."""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
df = pd.read_csv("predictive_maintenance.csv")
train, test = train_test_split(df, test_size=0.2, random_state=42)
x_train = train.drop(columns=["failure"])
y_train = train["failure"]
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(x_train[["vibration", "age", "avg_speed", "last_revision"]])
model = LogisticRegression()
model.fit(X_normalized, y_train)

"""

Réseau de neurone avec keras
On veut faire de la classification d'image (classer des images de machines) avec trois classes : 
machine en bon état
machine légèrement abimée
machine fortement abimée
On suppose que chaque image est représentée par un vecteur ayant 900 valeurs. Les labels ne
sont pas one-hot-encodés
Faire un réseau de neurones à 3 couches.
400 neurones pour la premiere
200 neurones pour la seconde
??? neurones pour la dernière couche
Compiler le modèle avec la bonne loss et l'optimizer “sgd


"""

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
model = Sequential()
model.add(Dense(400, activation='relu', input_shape=(900,)))
model.add(Dense(200, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.compile(optimizer=SGD(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# model.summary()  # Affiche le résumé du modèle





