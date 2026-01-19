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