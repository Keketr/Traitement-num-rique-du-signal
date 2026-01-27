"""Partie 1

1. Entraîner un réseau de neurone convolutionnel de petite taille (sans transfert learning) sur un dataset simple de type MNIST ou FashionMNIST jusqu'à atteindre une performance acceptable (sans rechercher la perfection).  Vous pouvez utiliser pytorch ou keras/tensorflow selon votre préférence."""
import numpy as np
import keras
from keras.models import load_model
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.utils import to_categorical

def train_cnn_model():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape((x_train.shape[0], 28, 28, 1)).astype('float32') / 255
    x_test = x_test.reshape((x_test.shape[0], 28, 28, 1)).astype('float32') / 255
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_test, y_test))

    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f'Test accuracy: {test_acc}')
    model.save('cnn_mnist_model.h5')
train_cnn_model()

def export_model_to_onnx(keras_model_path, onnx_model_path):
    model = load_model(keras_model_path)
    onnx_model = keras2onnx.convert_keras(model, model.name)
    keras2onnx.save_model(onnx_model, onnx_model_path)
export_model_to_onnx('cnn_mnist_model.h5', 'cnn_mnist_model.onnx')

"""2. Faire en sorte d'exporter le modèle au format ONNX. Si vous avez entraîné votre modèle avec pytorch vous aurez le sous module torch.onnx. Pour keras il existe https://github.com/onnx/tensorflow-onnx mais vous avez le droit d'utiliser votre moteur de recherche favori. """
import keras2onnx
from keras.models import load_model
def export_model_to_onnx(keras_model_path, onnx_model_path):
    model = load_model(keras_model_path)
    onnx_model = keras2onnx.convert_keras(model, model.name)
    keras2onnx.save_model(onnx_model, onnx_model_path)
export_model_to_onnx('cnn_mnist_model.h5', 'cnn_mnist_model.onnx')




""" Faire une fonction quantize(float_value, min_range, max_range, zero = 0) qui transforme un tableau numpy pouvant prendre des valeurs entre min_range et max_range en un tableau de int8.

On utilisera la formule 

q
u
a
n
t
i
z
e
d
=
I
n
t
(
f
l
o
a
t
_
v
a
l
u
e
/
S
)
quantized=Int(float_value/S)

où 

S
=
m
a
x
−
m
i
n
2
b
−
1
S= 
2 
b
 −1
max−min
​
  avec b le nombre de bit de la représentation visée

Remark : convert the numpy array type using .astype method"""


import numpy as np

def quantize(float_value, min_range, max_range, zero=0, bits=8):
    S = (max_range - min_range) / (2**bits - 1)
    quantized = (float_value / S)
    quantized = np.clip(quantized, -128, 127)
    return quantized.astype(np.int8)



"""  Faire une fonction to_float(uint_values, min_range, max_range) qui calcule l'opération inverse de la quantization de l'exercice précédent"""
def to_float(uint_values, min_range, max_range, bits=8):
    S = (max_range - min_range) / (2**bits - 1)
    float_values = uint_values.astype(np.float32) * S
    return float_values

"""récupérer les fonctions des exercices précédents et faire dans un notebook l'étude suivante : 

Faire une fonction display_error_histogram(values, min, max) qui affiche l'histogram des erreurs de quantization pour un schéma donnée. 

Faire une fonction quantization_error_stats(values, min, max) qui retourne la moyenne et l'écart type des erreurs de quantization d'un schéma donné. 

 

A l'aide des fonctions précédentes, étudier la quantization dans les situations suivantes : 

Situation 1

Un tableau de 1000 valeurs aléatoires uniformément distribuées entre 0 et 0.5
une quantization avec min_range = 0 et max_range = 1 """
import matplotlib.pyplot as plt
def display_error_histogram(values, min_range, max_range, bits=8):
    quantized_values = quantize(values, min_range, max_range, bits=bits)
    dequantized_values = to_float(quantized_values, min_range, max_range, bits=bits)
    errors = values - dequantized_values
    plt.hist(errors, bins=50)
    plt.title('Histogram of Quantization Errors')
    plt.xlabel('Error')
    plt.ylabel('Frequency')
    plt.show()
def quantization_error_stats(values, min_range, max_range, bits=8):
    quantized_values = quantize(values, min_range, max_range, bits=bits)
    dequantized_values = to_float(quantized_values, min_range, max_range, bits=bits)
    errors = values - dequantized_values
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    return mean_error, std_error
# Situation 1
values = np.random.uniform(0, 0.5, 1000)
min_range = 0
max_range = 1
display_error_histogram(values, min_range, max_range)
mean_error, std_error = quantization_error_stats(values, min_range, max_range)
print(f'Mean Error: {mean_error}, Std Error: {std_error}')

# pourquoi c'est intéressant ?
# Cette situation est intéressante car elle permet d'observer comment la quantization affecte les valeurs qui sont toutes situées dans la moitié inférieure de la plage spécifiée (0 à 1). Cela peut mettre en évidence les erreurs de quantization pour des valeurs faibles, ce qui est pertinent dans de nombreuses applications où les signaux ou données ont une amplitude limitée.


# Situation 1 bis 

"""Un tableau de 1000 valeurs aléatoires uniformément distribuées entre 0 et 1 avec S = 0 et S qui permet d'exploits l'entièreté des bits de la représentation entière
une quantization avec min_range = 0 et max_range = 1"""

values = np.random.uniform(0, 1, 1000)
min_range = 0
max_range = 1
display_error_histogram(values, min_range, max_range)
mean_error, std_error = quantization_error_stats(values, min_range, max_range)
print(f'Mean Error: {mean_error}, Std Error: {std_error}')

# pourquoi c'est intéressant ?
# Cette situation est intéressante car elle exploite pleinement la plage de quantization disponible pour les valeurs entre 0 et 1. En utilisant toute la gamme des bits disponibles, on peut minimiser les erreurs de quantization et observer comment cela affecte la précision des données quantifiées. Cela est particulièrement pertinent dans les applications où une haute fidélité est requise pour les signaux ou données numériques.


# Situation 2

"""Un tableau de 1000 valeurs aléatoires uniformément distribuées entre 0 et 2
une quantization avec min_range = 0 et max_range = 1"""

values = np.random.uniform(0, 2, 1000)
min_range = 0
max_range = 1
display_error_histogram(values, min_range, max_range)
mean_error, std_error = quantization_error_stats(values, min_range, max_range)
print(f'Mean Error: {mean_error}, Std Error: {std_error}')

# pourquoi c'est intéressant ?
# Cette situation est intéressante car elle met en évidence les effets de la quantization lorsque les valeurs d'entrée dépassent la plage spécifiée pour la quantization (0 à 1). Cela peut entraîner des erreurs significatives, notamment des saturations, où les valeurs supérieures à 1 sont toutes mappées à la valeur maximale quantifiée. Cela illustre l'importance de choisir une plage de quantization appropriée en fonction des caractéristiques des données d'entrée.


# Situation 3

""" Génère 1000 valeurs entre 100 et 101 Les quantifie avec min_range = 0 et max_range = 1000"""

values = np.random.uniform(100, 101, 1000)
min_range = 0
max_range = 1000
display_error_histogram(values, min_range, max_range)
mean_error, std_error = quantization_error_stats(values, min_range, max_range)
print(f'Mean Error: {mean_error}, Std Error: {std_error}')

# pourquoi c'est intéressant ?
# Cette situation est intéressante car elle examine comment la quantization affecte les valeurs qui sont toutes situées dans une plage très étroite (100 à 101) par rapport à la plage de quantization beaucoup plus large (0 à 1000). Cela peut entraîner des erreurs de quantization importantes, car les variations fines entre les valeurs proches de 100 peuvent être perdues lors de la quantization. Cela met en lumière les défis liés à la quantization de données avec une faible dynamique par rapport à la plage totale disponible.


"""Proposer une à 3 autres situation qui vous parait intéressante d'étudier et mener l'étude d'impact

Faire les etudes dans un notebook et rendre le notebook dans la section suivante"""

# Situation 4
"""Génère 1000 valeurs entre -1 et 1. Les quantifie avec min_range = -1 et max_range = 1"""
values = np.random.uniform(-1, 1, 1000)
min_range = -1
max_range = 1
display_error_histogram(values, min_range, max_range)
mean_error, std_error = quantization_error_stats(values, min_range, max_range)
print(f'Mean Error: {mean_error}, Std Error: {std_error}')

# pourquoi c'est intéressant ?
# Cette situation est intéressante car elle inclut des valeurs négatives, ce qui permet d'observer comment la quantization gère les plages de valeurs symétriques autour de zéro. Cela peut être pertinent pour des applications où les données peuvent varier dans les deux directions, comme les signaux audio ou les images avec des valeurs de pixel négatives après certaines transformations.

# Situation 5


"""Génère 1000 valeurs entre 0 et 10 avec une distribution normale centrée autour de 5. Les quantifie avec min_range = 0 et max_range = 10"""
values = np.random.normal(5, 2, 1000)
values = np.clip(values, 0, 10)  # Assurer que les valeurs restent dans la plage [0, 10]
min_range = 0
max_range = 10
display_error_histogram(values, min_range, max_range)
mean_error, std_error = quantization_error_stats(values, min_range, max_range)
print(f'Mean Error: {mean_error}, Std Error: {std_error}')
# pourquoi c'est intéressant ?
# Cette situation est intéressante car elle utilise une distribution normale, ce qui est courant dans de nombreuses applications réelles. En quantifiant des données qui ne sont pas uniformément réparties, on peut observer comment la quantization affecte les valeurs proches de la moyenne par rapport aux valeurs aux extrémités de la distribution.


# Situation 6
"""Génère 1000 valeurs entre 0 et 1 avec une distribution exponentielle. Les quantifie avec min_range = 0 et max_range = 1"""
values = np.random.exponential(0.2, 1000)
values = np.clip(values, 0, 1)  # Assurer que les valeurs restent dans la plage [0, 1]
min_range = 0
max_range = 1
display_error_histogram(values, min_range, max_range)
mean_error, std_error = quantization_error_stats(values, min_range, max_range)
print(f'Mean Error: {mean_error}, Std Error: {std_error}')
# pourquoi c'est intéressant ?
# Cette situation est intéressante car la distribution exponentielle est souvent utilisée pour modéliser des phénomènes tels que les temps d'attente ou les durées de vie. En quantifiant des données avec une telle distribution, on peut évaluer comment la quantization affecte les valeurs qui sont plus concentrées vers zéro, ce qui peut être crucial pour certaines applications statistiques ou de fiabilité.



"""
Partie 0  - quantization naive

 
Entraîner un MLP simple avec pytorch sur mnist ou tout autre dataset simple
Faire une fonction quantize(model) qui effectue une copie du modèle et passe tous les poids en int naivement.
Tester votre modèle quantifié et comparer l'accuracy au modèle original. Cela marche-t-il ?"""

import torch
import copy
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
def quantize_model(model):
    q = copy.deepcopy(model)
    with torch.no_grad():
        for p in q.parameters():
            p.copy_(torch.round(p))  # valeurs entières, dtype float conservé
    return q
def train_and_evaluate():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    model = SimpleMLP()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(5):
        model.train()
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    original_accuracy = correct / len(test_loader.dataset)
    print(f'Original Model Accuracy: {original_accuracy}')

    quantized_model = quantize_model(model)
    quantized_model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = quantized_model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    quantized_accuracy = correct / len(test_loader.dataset)
    print(f'Quantized Model Accuracy: {quantized_accuracy}')
train_and_evaluate()
# Pourquoi cela ne marche pas ?
# La quantization naïve des poids en entiers peut entraîner une perte significative d'information, car les poids originaux en virgule flottante peuvent contenir des valeurs très précises qui sont arrondies ou tronquées lors de la conversion en entiers. Cela peut affecter la capacité du modèle à apprendre et à généraliser correctement, entraînant une baisse de performance, comme observé dans l'accuracy du modèle quantifié par rapport au modèle original.



"""Partie 1 - quantization statique - weights only

 

Vous voulez désormais faire en sorte de quantifier votre modèle en quantifiant uniquement les poids avec le schéma de quantification précédent. Pour cela vous aller quantifier chaque layer de votre modèle avec des plages différentes. 

Dans un premier temps calculer le min et max de tous les paramètre de votre réseau et faire en sorte de quantifier les paramètres avec une unique plage pour toutes les couches.
Comparer les performances du modèle
Faire pareil mais en faisant en sorte de quantifier chaque couche avec sa propre plage de valeur.
Cela marche-t-il ? Sinon que suggérez vous pour résoudre le problème ?"""
import torch
import copy
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
def quantize_model_global(model):
    q = copy.deepcopy(model)
    all_params = torch.cat([p.view(-1) for p in q.parameters()])
    min_val = all_params.min()
    max_val = all_params.max()
    S = (max_val - min_val) / 255.0
    with torch.no_grad():
        for p in q.parameters():
            p.copy_(torch.round((p - min_val) / S))
    return q
def quantize_model_per_layer(model):
    q = copy.deepcopy(model)
    with torch.no_grad():
        for p in q.parameters():
            min_val = p.min()
            max_val = p.max()
            S = (max_val - min_val) / 255.0
            p.copy_(torch.round((p - min_val) / S))
    return q
def train_and_evaluate():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    model = SimpleMLP()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(5):
        model.train()
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    original_accuracy = correct / len(test_loader.dataset)
    print(f'Original Model Accuracy: {original_accuracy}')

    quantized_model_global = quantize_model_global(model)
    quantized_model_global.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = quantized_model_global(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    quantized_global_accuracy = correct / len(test_loader.dataset)
    print(f'Quantized Model (Global) Accuracy: {quantized_global_accuracy}')

    quantized_model_per_layer = quantize_model_per_layer(model)
    quantized_model_per_layer.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = quantized_model_per_layer(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    quantized_per_layer_accuracy = correct / len(test_loader.dataset)
    print(f'Quantized Model (Per Layer) Accuracy: {quantized_per_layer_accuracy}')
train_and_evaluate()