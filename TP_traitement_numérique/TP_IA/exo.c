/**
 * 
 * @file exo.c
 * 
 * codage d'exo en C
 */



 /**
  

  Exo 1 : 1. Écrire une fonction fill_vector qui va créer un vecteur de type float et de taille N. Elle doit avoir le prototype suivant : 

float*    fill_vector(int N);

2. Remplir le vecteur avec des valeurs allant de 1 jusqu'à N. 
 

Exemple : 

fill_vector(5) doit renvoyer le tableau suivant : {1, 2, 3, 4, 5}
  */


#include <stdio.h>
#include <stdlib.h>

float* fill_vector(int N) {
    // Allocation dynamique d'un tableau de float de taille N
    float* vector = (float*)malloc(N * sizeof(float));
    if (vector == NULL) {
        // Gestion de l'erreur d'allocation mémoire
        fprintf(stderr, "Erreur d'allocation mémoire\n");
        return NULL;
    }

    // Remplissage du tableau avec des valeurs de 1 à N
    for (int i = 0; i < N; i++) {
        vector[i] = (float)(i + 1);
    }

    return vector;
}

int main() {
    int N = 5;
    float* vector = fill_vector(N);
    
    if (vector != NULL) {
        // Affichage du vecteur pour vérification
        for (int i = 0; i < N; i++) {
            printf("%.1f ", vector[i]);
        }
        printf("\n");

        // Libération de la mémoire allouée
        free(vector);
    }

    return 0;
}


/*

//////////////////////////////////

EXO 2 :

1. Écrire une fonction create_matrice qui va créer une matrice. Le nombre de lignes (nbRow) et de colonnes (nbColumn) seront donnés en paramètre.

Elle doit avoir le prototype suivant :

int** createMatrice(int nbRow, int nbColumn);

2. Remplir la matrice de la valeur 0.

 

Exemple :

createMatrice(2, 4) doit renvoyer :

new_matrice[2][4] = {

{0, 0, 0, 0},

{0, 0, 0, 0}

};

/////////////////////////////////




*/

#include <stdio.h>
#include <stdlib.h>
int** createMatrice(int nbRow, int nbColumn) {
    // Allocation dynamique d'un tableau de pointeurs pour les lignes
    int** matrice = (int**)malloc(nbRow * sizeof(int*));
    if (matrice == NULL) {
        fprintf(stderr, "Erreur d'allocation mémoire pour les lignes\n");
        return NULL;
    }

    // Allocation dynamique pour chaque ligne
    for (int i = 0; i < nbRow; i++) {
        matrice[i] = (int*)malloc(nbColumn * sizeof(int));
        if (matrice[i] == NULL) {
            fprintf(stderr, "Erreur d'allocation mémoire pour les colonnes\n");
            // Libération de la mémoire déjà allouée avant de retourner
            for (int j = 0; j < i; j++) {
                free(matrice[j]);
            }
            free(matrice);
            return NULL;
        }
    }

    // Remplissage de la matrice avec des zéros
    for (int i = 0; i < nbRow; i++) {
        for (int j = 0; j < nbColumn; j++) {
            matrice[i][j] = 0;
        }
    }

    return matrice;
}


/*

//////////////////////////////////

EXO 3 :

Create a function with the following prototype:

float trace(float** matrix, int n_row, int n_col)

that calculates the trace of a matrix

You only need to code the function, not the main.


/////////////////////////////////



*/


#include <stdio.h>
float trace(float** matrix, int n_row, int n_col) {
    float trace_sum = 0.0f;
    int min_dim = (n_row < n_col) ? n_row : n_col; // La trace est définie pour les matrices carrées

    for (int i = 0; i < min_dim; i++) {
        trace_sum += matrix[i][i];
    }

    return trace_sum;
}



/*

//////////////////////////////////

EXO 4 :
Create a function int* matrice_multiplication(int** matrice, int* vecteur, int vector_size) that:

1. Takes as parameters a square matrix (matrice), a vector (vecteur), and the size of the vector (vector_size). It returns the result of multiplying the matrix by the vector.  Assumes the matrix is square and its dimensions match the vector size

2. Test your function using test_main

/////////////////////////////////

*/


#include <stdio.h>
int* matrice_multiplication(int** matrice, int* vecteur, int vector_size) {
    // Allocation dynamique pour le vecteur résultat
    int* result = (int*)malloc(vector_size * sizeof(int));
    if (result == NULL) {
        fprintf(stderr, "Erreur d'allocation mémoire pour le vecteur résultat\n");
        return NULL;
    }

    // Initialisation du vecteur résultat à zéro
    for (int i = 0; i < vector_size; i++) {
        result[i] = 0;
    }

    // Multiplication de la matrice par le vecteur
    for (int i = 0; i < vector_size; i++) {
        for (int j = 0; j < vector_size; j++) {
            result[i] += matrice[i][j] * vecteur[j];
        }
    }

    return result;
}

void test_main() {
    int vector_size = 3;

    // Exemple de matrice 3x3
    int** matrice = (int**)malloc(vector_size * sizeof(int*));
    for (int i = 0; i < vector_size; i++) {
        matrice[i] = (int*)malloc(vector_size * sizeof(int));
    }

    // Remplissage de la matrice
    int count = 1;
    for (int i = 0; i < vector_size; i++) {
        for (int j = 0; j < vector_size; j++) {
            matrice[i][j] = count++;
        }
    }

    // Exemple de vecteur
    int vecteur[] = {1, 2, 3};

    // Appel de la fonction de multiplication
    int* result = matrice_multiplication(matrice, vecteur, vector_size);

    // Affichage du résultat
    printf("Résultat de la multiplication matrice-vecteur:\n");
    for (int i = 0; i < vector_size; i++) {
        printf("%d\n", result[i]);
    }

    // Libération de la mémoire
    for (int i = 0; i < vector_size; i++) {
        free(matrice[i]);
    }
    free(matrice);
    free(result);
}



/*Coder une fonction ReLu(x) qui calcule la fonction relu
Coder une fonction relu_no_if(x) qui calcule la fonction relu sans utiliser de conditions
Sur votre pc : faire un benchmarking des deux fonctions et voir si une des fonctions s exécute plus vite que l autre. 
pour chaque fonction vous ferez une boucle for qui permet de calculer le temps moyen sur un grand nombre de fois*/

#include <stdio.h>
#include <time.h>
float ReLu(float x) {
    if (x < 0) {
        return 0;
    } else {
        return x;
    }
}
float relu_no_if(float x) {
    return (x + fabs(x)) / 2.0f;
    // pourquoi on divise par 2 ? parce que pour x positif, (x + x)/2 = x, et pour x négatif, (x + (-x))/2 = 0
   
}
void benchmark() {
    const int iterations = 1000000;
    float test_value = -5.0f;

    
    clock_t start_if = clock();
    for (int i = 0; i < iterations; i++) {
        ReLu(test_value);
    }
    clock_t end_if = clock();
    double time_if = (double)(end_if - start_if) / CLOCKS_PER_SEC;


    clock_t start_no_if = clock();
    for (int i = 0; i < iterations; i++) {
        relu_no_if(test_value);
    }
    clock_t end_no_if = clock();
    double time_no_if = (double)(end_no_if - start_no_if) / CLOCKS_PER_SEC;

    printf("Time taken by ReLu with if: %f seconds\n", time_if);
    printf("Time taken by relu_no_if: %f seconds\n", time_no_if);
}

/*

Coder une fonction float* softmax(float *values, int n_values) qui normalise par la fonction softmax un vecteur de fonction d'activitation. 

Vous utiliserez la fonction exp_approx fournie avec avec n = 10;

*/

#include <stdio.h>
#include <math.h>
float exp_approx(float x, int n) {
    float sum = 1.0f; // e^0 = 1
    float term = 1.0f; // Premier terme de la série

    for (int i = 1; i <= n; i++) {
        term *= x / i; // Calcul du terme suivant
        sum += term;   // Ajout du terme à la somme
    }

    return sum;
}

float* softmax(float *values, int n_values) {
    float* exp_values = (float*)malloc(n_values * sizeof(float));
    float sum_exp = 0.0f;

    
    for (int i = 0; i < n_values; i++) {
        exp_values[i] = exp_approx(values[i], 10);
        sum_exp += exp_values[i];
    }


    for (int i = 0; i < n_values; i++) {
        exp_values[i] /= sum_exp;
    }

    return exp_values;
}



/*

Coder une fonction relu_neurone(float *values, float* weights int n_features) qui calcule la sortie d'un neurone sur une entrée

Le tableau de weights contiendra un poids de plus que le tableau de values pour tenir compte du biais. 

La fonction ReLu de l'exercice précédent est disponible

*/

#include <stdio.h>
float relu_neurone(float *values, float *weights, int n_features)
{
    float sum = weights[0]; // biais

    for (int i = 0; i < n_features; i++) {
        sum += values[i] * weights[i + 1];
    }

    return ReLu(sum);
}


/*

Coder une fonction compute_layer_output(float *values, float **weights, int n_feature, int n_neurone) qui calcule la sortie d'une couche contanant n_neurones et dont les poids sont stockés dans weights.

 

Vous pourrez réutiliser la fonction relu_neurone pour cela

*/

#include <stdio.h>
float* compute_layer_output(float *values, float **weights, int n_feature, int n_neurone) {
    float* output = (float*)malloc(n_neurone * sizeof(float));

    for (int i = 0; i < n_neurone; i++) {
        output[i] = relu_neurone(values, weights[i], n_feature);
    }

    return output;
}


/*

Coder une fonction two_layer_network(float *features, int n_feature) qui prend entrée une donnée avec n_feature et qui calcule la prédiction du réseau. 

Le réseau aura deux couches : 

1 couche feedforward avec 5 neurones (relu)
1 couche feedforward avec 2 neurones (softmax)
Les weights seront défini via des tableaux dans la fonction two_layer_network

Pour tester votre réseau vous pourrez entraîner un modèle en keras sur des données aléatoire gaussien à 2 ou 3 dimensions. 

Pour les labels  on considerera que y = 1 si x_1 + x_2 + x_3 > 0 

y = 0 sinon.

Vous vérifierez alors que votre modèle en C fait les même prédiction que le modèle keras

*/
#include <stdio.h>
float* two_layer_network(float *features, int n_feature) {
    float weights_layer1[5][4] = {
        {0.2f, 0.4f, -0.5f, 0.1f},
        {-0.3f, 0.2f, 0.6f, -0.1f},
        {0.5f, -0.4f, 0.3f, 0.2f},
        {0.1f, 0.3f, -0.2f, 0.4f},
        {-0.2f, 0.5f, 0.1f, -0.3f}
    };

    float *layer1_output = compute_layer_output(features, &weights_layer1[0][0], n_feature, 5);

    float weights_layer2[2][6] = {
        {0.3f, -0.2f, 0.4f, 0.1f, -0.5f, 0.2f},
        {-0.4f, 0.3f, -0.1f, 0.5f, 0.2f, -0.3f}
    };

    float *layer2_output = compute_layer_output(layer1_output, &weights_layer2[0][0], 5, 2);

    float *final_output = softmax(layer2_output, 2);

 

    return final_output;

/*

On veut faire un petit réseau de neurone convolutionnel 1D en C

Pour cela, coder la fonction float* convolution_1d(float *data, float *kernel, int n_values, int kernel_size)

qui permet de calculer la convolution d'un signal par un kernel 1D. 

*/

#include <stdio.h>
float* convolution_1d(float *data, float *kernel, int n_values, int kernel_size) {
    int output_size = n_values - kernel_size + 1;
    float* output = (float*)malloc(output_size * sizeof(float));

    for (int i = 0; i < output_size; i++) {
        output[i] = 0.0f;
        for (int j = 0; j < kernel_size; j++) {
            output[i] += data[i + j] * kernel[j];
        }
    }

    return output;
}


/*

En utilisant la fonction de l'exercice précédent coder une fonction 

float ** compute_layer_output(float *data,  float **kernels, int n_values, int kernel_size)

qui retourne un tableau 2D de float où chaque ligne correspond au passage d'un kerner l sur le signal original. 

*/

#include <stdio.h>
float** compute_layer_output(float *data, float **kernels, int n_values, int kernel_size, int n_kernels) {
    int output_size = n_values - kernel_size + 1;
    float** output = (float**)malloc(n_kernels * sizeof(float*));

    for (int k = 0; k < n_kernels; k++) {
        output[k] = (float*)malloc(output_size * sizeof(float));
        for (int i = 0; i < output_size; i++) {
            output[k][i] = 0.0f;
            for (int j = 0; j < kernel_size; j++) {
                output[k][i] += data[i + j] * kernels[k][j];
            }
        }
    }

    return output;
}


/*

Coder la fonction convolution_2d(float ** values, int n_row, n_col, int kernel_size)

on supposera que 

le kernel est une matrice carré
stride = 1
pas de padding 

*/

#include <stdio.h>
float** convolution_2d(float ** values, int n_row, int n_col, float ** kernel, int kernel_size) {
    int output_rows = n_row - kernel_size + 1;
    int output_cols = n_col - kernel_size + 1;
    float** output = (float**)malloc(output_rows * sizeof(float*));

    for (int i = 0; i < output_rows; i++) {
        output[i] = (float*)malloc(output_cols * sizeof(float));
        for (int j = 0; j < output_cols; j++) {
            output[i][j] = 0.0f;
            for (int ki = 0; ki < kernel_size; ki++) {
                for (int kj = 0; kj < kernel_size; kj++) {
                    output[i][j] += values[i + ki][j + kj] * kernel[ki][kj];
                }
            }
        }
    }

    return output;
}

