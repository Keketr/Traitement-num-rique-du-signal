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

Coder une fonction float linear_regression_prediction(float* features, float* thetas, int n_parameters) qui calcul la prediction d'une regression lineaire pour une donnée passée dans features

Exemple d'entrée 

X = [1, 1, 1]

theta = [0, 1, 1, 1]

Pas de fonction main, utiliser test_main

*/



float linear_regression_prediction(float* features, float* thetas, int n_parameters) {
    

    float pred = thetas[0];
    for (int i = 1; i < n_parameters; ++i)
    {
        pred += thetas[i] * features[i - 1];
    }
    return pred;
}


void test_main(){
    float features[] = {1.0f, 1.0f, 1.0f};
    float thetas[]   = {0.0f, 1.0f, 1.0f, 1.0f};

    float y = linear_regression_prediction(features, thetas, 4);
    printf("%f\n", y); 
}


/*

Coder une fonction logistic_regression(float* features, float* thetas, int n_parameter) qui calcul la prediction d'une regression logistique.

 

Attention thetas contient une valeur de plus que features pour tenir compte du bias (theta0). Le bias est stocké à l'indice 0 dans thetas

La fonction sigmoid est mise disposition (le code est caché)

*/


float logistic_regression(float* features, float* thetas, int n_parameter) {
    
    float z = thetas[0];
    for (int i = 1; i < n_parameter; ++i) {
        z += thetas[i] * features[i - 1];
    }
    return sigmoid(z);
}

void test_main(){
    
    float f1[] = {1.0f, 1.0f, 1.0f};
    float t1[] = {0.0f, 1.0f, 1.0f, 1.0f};
    printf("test1: %f\n", logistic_regression(f1, t1, 4));
    
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
    // pourquoi on fait ça ? Parce que si x est négatif, fabs(x) le rend positif, donc la somme est nulle. Si x est positif, la somme est 2*x, et en divisant par 2 on obtient x.
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



*/

#include <stdio.h>
float* two_layer_network(float *features, int n_feature) {
    // Définition des poids pour la première couche (5 neurones)
    float weights_layer1[5][n_feature + 1] = {
        {0.2, 0.4, -0.5, 0.1}, // Poids pour le neurone 1
        {-0.3, 0.8, 0.6, -0.2}, // Poids pour le neurone 2
        {0.5, -0.7, 0.3, 0.4}, // Poids pour le neurone 3
        {0.1, 0.2, -0.4, 0.9}, // Poids pour le neurone 4
        {-0.6, 0.3, 0.7, -0.1}  // Poids pour le neurone 5
    };

    // Calcul de la sortie de la première couche
    float* layer1_output = compute_layer_output(features, (float**)weights_layer1, n_feature, 5);

    // Définition des poids pour la deuxième couche (2 neurones)
    float weights_layer2[2][5 + 1] = {
        {0.3, -0.2, 0.5, 0.1, -0.4, 0.6}, // Poids pour le neurone 1
        {-0.5, 0.4, -0.3, 0.7, 0.2, -0.1} // Poids pour le neurone 2
    };

    // Calcul de la sortie de la deuxième couche
    float* layer2_output = compute_layer_output(layer1_output, (float**)weights_layer2, 5, 2);

    // Application de la fonction softmax sur la sortie de la deuxième couche
    float* final_output = softmax(layer2_output, 2);

    // Libération de la mémoire allouée pour les sorties intermédiaires
    free(layer1_output);
    free(layer2_output);

    return final_output;
}

/*

Pour tester votre réseau vous pourrez entraîner un modèle en keras sur des données aléatoire gaussien à 2 ou 3 dimensions. 

Pour les labels  on considerera que y = 1 si x_1 + x_2 + x_3 > 0 

y = 0 sinon.

Vous vérifierez alors que votre modèle en C fait les même prédiction que le modèle keras

*/




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

// test main pour convolution_1d
void test_convolution_1d() {
    float data[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    float kernel[] = {0.2, 0.5, 0.3};
    int n_values = 5;
    int kernel_size = 3;

    float* result = convolution_1d(data, kernel, n_values, kernel_size);
    int output_size = n_values - kernel_size + 1;

    printf("Résultat de la convolution 1D:\n");
    for (int i = 0; i < output_size; i++) {
        printf("%.2f ", result[i]);
    }
    printf("\n");

    free(result);
}

// résultat 

/*

Descente gradient C
On veut faire la descente de gradient sur une fonction f: x -> y qui renvoie un scalaire.
La fonction deriv_f qui calcule la dérivée de f en x est mise à votre disposition (le code est caché)
Code en C la fonction descente_gradient qui prend en entrée un x initial, un nombre d'itération et
un learning rate
dans la fonction test_main, appliquer la descente de gradient sur 20 itérations avec un learning
rate de 0.1 et un x_init= 2.5

*/

#include <stdio.h>
float deriv_f(float x); // Prototype de la fonction dérivée (implémentation cachée)
float descente_gradient(float x_init, int n_iterations, float learning_rate) {
    float x = x_init;

    for (int i = 0; i < n_iterations; i++) {
        float grad = deriv_f(x);
        x = x - learning_rate * grad;
    }

    return x;
}
void test_main() {
    float x_init = 2.5f;
    int n_iterations = 20;
    float learning_rate = 0.1f;

    float x_optimized = descente_gradient(x_init, n_iterations, learning_rate);
    printf("Valeur optimisée de x après descente de gradient: %f\n", x_optimized);
}

/*

Descente de gradient C - la valeur convergée
en faisant tourner l'algorithme de descente de gradient suffisament longtemps, indiquer quel est à
votre avis la valeur numérique du argmin de la fonction cachée

*/
// La valeur numérique du argmin de la fonction cachée est approximativement 1.0 après une descente de gradient suffisante.






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
        output[k] = convolution_1d(data, kernels[k], n_values, kernel_size);
    }

    return output;
}


// test main pour compute_layer_output
void test_compute_layer_output() {
    float data[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    int n_values = 5;
    int kernel_size = 3;
    int n_kernels = 2;

    // Définition de deux kernels
    float** kernels = (float**)malloc(n_kernels * sizeof(float*));
    kernels[0] = (float[]){0.2, 0.5, 0.3};
    kernels[1] = (float[]){-0.1, 0.4, 0.6};

    float** result = compute_layer_output(data, kernels, n_values, kernel_size, n_kernels);
    int output_size = n_values - kernel_size + 1;

    printf("Résultat de la couche de convolution:\n");
    for (int k = 0; k < n_kernels; k++) {
        printf("Kernel %d: ", k + 1);
        for (int i = 0; i < output_size; i++) {
            printf("%.2f ", result[k][i]);
        }
        printf("\n");
    }

    // Libération de la mémoire
    for (int k = 0; k < n_kernels; k++) {
        free(result[k]);
    }
    free(result);
    free(kernels);
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

/*

Créer une fonction find_first_value qui retourne l'indice de la première valeur recherchée. Elle doit renvoyée -1 si la valeur n'est pas dans le tableau

La fonction doit  avoir le prototype suivant : 

int find_first_value (float value_to_find, float* values, int n_values) 

Remarque : pas besoin de fonction main ! 

*/

#include <stdio.h>
int find_first_value (float value_to_find, float* values, int n_values) {
    for (int i = 0; i < n_values; i++) {
        if (values[i] == value_to_find) {
            return i; // Retourne l'indice de la première occurrence
        }
    }
    return -1; // Valeur non trouvée
}


/*

Créer une fonction avant le prototype suivant : 

float* array_filter_below(float values[], int n_values, float max_value) 

qui créer un tableau ne contenant les valeurs de  values inferieure à max_values. 

 

Vous n'avez pas besoin de coder le main, seulement la fonction. 

*/

#include <stdio.h>
#include <stdlib.h>
float* array_filter_below(float values[], int n_values, float max_value) {
    // Première passe pour compter le nombre d'éléments inférieurs à max_value
    int count = 0;
    for (int i = 0; i < n_values; i++) {
        if (values[i] < max_value) {
            count++;
        }
    }

    // Allocation du tableau résultat
    float* filtered_array = (float*)malloc(count * sizeof(float));
    

    // Deuxième passe pour remplir le tableau résultat
    int index = 0;
    for (int i = 0; i < n_values; i++) {
        if (values[i] < max_value) {
            filtered_array[index++] = values[i];
        }
    }

    
    return filtered_array;
}


/*

Créer une fonction nommée int** add_matrices(int ** mat1, int **mat2, int n_row, int n_col) qui calcule la somme des matrices mat1 et mat2

Il faudra créer une matrice m3 avec malloc de la bonne taille et retourner le pointeur

*/

#include <stdio.h>
#include <stdlib.h>
int** add_matrices(int ** mat1, int **mat2, int n_row, int n_col) {
    // Allocation dynamique pour la matrice résultat
    int** result = (int**)malloc(n_row * sizeof(int*));
    for (int i = 0; i < n_row; i++) {
        result[i] = (int*)malloc(n_col * sizeof(int));
    }

    // Calcul de la somme des matrices
    for (int i = 0; i < n_row; i++) {
        for (int j = 0; j < n_col; j++) {
            result[i][j] = mat1[i][j] + mat2[i][j];
        }
    }

    return result;
}


/*

Créer une fonction accuracy(int* y, int* y_preds, int n_values) qui calcule l'accuracy (précision) d'un modèle de prédiction.

L'accuracy est définie comme la proportion de prédictions correctes parmi toutes les prédictions. Une prédiction est considérée comme correcte lorsque la valeur prédite (y_preds[i]) est égale à la valeur réelle (y[i]).

La fonction doit retourner un float représentant cette proportion (entre 0.0 et 1.0).

*/


#include <stdio.h>
float accuracy(int* y, int* y_preds, int n_values) {
    int correct_predictions = 0;

    for (int i = 0; i < n_values; i++) {
        if (y[i] == y_preds[i]) {
            correct_predictions++;
        }
    }

    return (float)correct_predictions / n_values;
}


/*

En effectuant des recherches si besoin, faire expliquer avec vos propres mots ce que c'est qu'avec des classes déséquilibrées dans son dataset (class imbalances)
Une classe déséquilibrée dans un dataset fait référence à une situation où les différentes classes ou catégories d'un ensemble de données ne sont pas représentées de manière égale. 
Par exemple, dans un problème de classification binaire, si 90% des exemples appartiennent à la classe A et seulement 10% à la classe B, on dit que les classes sont déséquilibrées.

*/


/*

On suppose qu'on a un dataset avec 1000 lignes.
Pour 900 lignes y = 0

et 100 lignes y = 1. 

Indiquer quelle serait l'accuracy (entre 0  et 1) d'un modèle  f_w: x → 0 

*/

//La précision (accuracy) d'un modèle qui prédit toujours y = 0 dans un dataset où 900 lignes ont y = 0 et 100 lignes ont y = 1 serait de 0,9 (ou 90%).


/*

En effectuant des recherches si besoin, faire expliquer pourquoi l'accuracy peut etre un faux ami dans le cas où on a un dataset déséquilibré. 

The answer correctly identifies that accuracy can be misleading with imbalanced datasets and mentions how a model might simply predict the majority class. 
However, the explanation lacks depth and specific details - no mention of metrics better suited for imbalanced data (like precision, recall, F1-score), no numerical examples to illustrate the point, and no discussion of why this is problematic from a practical perspective. 
The response is coherent but brief, missing opportunities to demonstrate a more thorough understanding of the issue.

*/


/*

En faitsant des recherches si besoin, expliquer avec vos propres mots ce qu'est la précision et le recall en machine learning. 
La précision (precision) en machine learning est une métrique qui mesure la proportion de prédictions positives correctes parmi toutes les prédictions positives effectuées par le modèle.
Le rappel (recall), également connu sous le nom de sensibilité, mesure la proportion de vraies positives qui ont été correctement identifiées par le modèle parmi toutes les instances positives réelles dans le dataset.
 

*/


/*

Expliquer ce qu'est une matrice de confusion et ce qu'elle permet de mesurer
Une matrice de confusion est un tableau utilisé pour évaluer la performance d'un modèle de classification. 
Elle présente les prédictions du modèle par rapport aux valeurs réelles, permettant ainsi de visualiser les erreurs de classification. 
Elle permet de mesurer des métriques telles que la précision, le rappel, la spécificité et le F1-score, offrant une compréhension plus approfondie de la performance du modèle au-delà de l'accuracy simple.


*/