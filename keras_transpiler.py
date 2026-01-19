"""
Faire un script python keras_transpiler qui charge un modèle keras, parcours ses couches et génère un code C permettant de faire la prédiction du modèle sur une donnée. 

Les poids du modèles devront être hardcodés dans le fichier C"""


import keras
import numpy as np
from keras.models import load_model
def keras_to_c(model_path, output_c_path):
    model = load_model(model_path)
    with open(output_c_path, 'w') as f:
        f.write('#include <stdio.h>\n\n')
        f.write('float predict(float input[]) {\n')
        f.write('    float layer_output[{}];\n'.format(model.layers[0].output_shape[1]))
        
        for i, layer in enumerate(model.layers):
            weights = layer.get_weights()
            if len(weights) == 0:
                continue
            
            weight_matrix = weights[0]
            bias_vector = weights[1]
            input_size = weight_matrix.shape[0]
            output_size = weight_matrix.shape[1]
            
            f.write('    // Layer {}\n'.format(i))
            f.write('    for (int j = 0; j < {}; j++) {{\n'.format(output_size))
            f.write('        layer_output[j] = 0;\n')
            f.write('        for (int k = 0; k < {}; k++) {{\n'.format(input_size))
            f.write('            layer_output[j] += input[k] * {:.6f};\n'.format(weight_matrix[k][j]))
            f.write('        }\n')
            f.write('        layer_output[j] += {:.6f};\n'.format(bias_vector[j]))
            if layer.activation.__name__ == 'relu':
                f.write('        if (layer_output[j] < 0) layer_output[j] = 0;\n')
            f.write('    }\n')
            f.write('    for (int m = 0; m < {}; m++) {{\n'.format(output_size))
            f.write('        input[m] = layer_output[m];\n')
            f.write('    }\n\n')
        
        f.write('    return layer_output[0]; // Assuming single output\n')
        f.write('}\n')


