#include <stdio.h>
#include <stdlib.h>

#define input_size 39
#define hidden_size1 100
#define output_size 10

double* bias = NULL;
double* weight1 = NULL;
double* weight2 = NULL;
double* input = NULL;


double* forward();
void vector_add(double* vector, double* bias);
double* matrix_mul(unsigned int num_row, unsigned int num_col, double* matrix,
                 double* vector);
void initialization(double* matrix, unsigned int size, char* name);

int main() {

  input = (double*)malloc(input_size*sizeof(double));
  bias = (double*)malloc(hidden_size1*sizeof(double));
  weight1 = (double*)malloc(input_size*hidden_size1*sizeof(double));
  weight2 = (double*)malloc(hidden_size1*output_size*sizeof(double));

  initialization(input, input_size, "temp2.txt");
  initialization(bias, hidden_size1, "bias.txt");
  initialization(weight1, input_size * hidden_size1, "weight1.txt");
  initialization(weight2, hidden_size1*output_size, "weight2.txt");

  double* result = forward();

  printf("result: \n");
  for (int i = 0; i < output_size; i++) {
    printf("%f ", result[i]);
  }
  printf("\n");

  free(input);
  free(bias);
  free(weight1);
  free(weight2);
  free(result);

  return 0;
}


void initialization(double* matrix, unsigned int size, char* name) {
  //double* matrix[size];
  int i=0;
  FILE *fp;
  char str[100];
  fp = fopen(name, "r");
  if(fp == NULL) {
    perror("Error opening file");
    exit(1);
  }
  while (i < size) {
    /* writing content to stdout */
    fgets (str, 100, fp);

    matrix[i] = atof(str);
    //printf("%.16g ", data[i]);
    i++;
  }
  /*for (int j = 0; j < size; j++) {
    printf("%.16g ", matrix[j]);
  }*/
  fclose(fp);

}


double* forward() {
  double* vector2 = matrix_mul(hidden_size1, input_size,  weight1, input);
  vector_add(vector2, bias);
  double* output = (double*)malloc(output_size*sizeof(double));
  output = matrix_mul(output_size, hidden_size1, weight2, vector2);
  free(vector2);

  return output;
}

void vector_add(double* vector, double* bias) {
  for (int i = 0; i < hidden_size1; i++) {
    vector[i] = vector[i] + bias[i];
  }
}

double* matrix_mul(unsigned int num_row, unsigned int num_col, double* matrix,
                 double* vector) {

  double sum = 0.0;
  double* value = (double*)malloc(num_col*sizeof(double));

  for (int row = 0; row < num_row; row++) {
    for (int col = 0; col < num_col; col++) {
      sum += matrix[row * num_col + col] * vector[col];
    }
    if (sum > 0) {
      value[row] = sum;
    } else {
      value[row] = 0.0;
    }
    sum = 0;
    //printf("%f ", value[row]);
  }
  //printf("\n");
  return value;
}
