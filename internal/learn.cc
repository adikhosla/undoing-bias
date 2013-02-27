/*
 * Optimize LSVM objective function via gradient descent.
 *
 * We use an adaptive cache mechanism.  After a negative example
 * scores beyond the margin multiple times it is removed from the
 * training set for a fixed number of iterations.
 * 
 * Originally included in voc-release3
 *
 * Modified by Aditya Khosla for ECCV 2012 paper:
 * Undoing the Damage of Dataset Bias
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <errno.h>

// Data File Format
// EXAMPLE*
// 
// EXAMPLE:
//  long label          ints
//  blocks              int
//  dim                 int
//  DATA{blocks}
//
// DATA:
//  block label         float
//  block data          floats
//
// Internal Binary Format
//  len           int (byte length of EXAMPLE)
//  EXAMPLE       <see above>
//  unique flag   byte

// number of iterations
#define ITER 5000000

// small cache parameters
#define INCACHE 3
#define WAIT 10

// error checking
#define check(e) \
(e ? (void)0 : (printf("%s:%u error: %s\n%s\n", __FILE__, __LINE__, #e, strerror(errno)), exit(1)))

// number of non-zero blocks in example ex
#define NUM_NONZERO(ex) (((int *)ex)[labelsize+1])

// float pointer to data segment of example ex
#define EX_DATA(ex) ((float *)(ex + sizeof(int)*(labelsize+3)))

// class label (+1 or -1) for the example
#define LABEL(ex) (((int *)ex)[1])

// block label (converted to 0-based index)
#define BLOCK_IDX(data) (((int)data[0])-1)

// bias label for the examples (converted to 0-based)
#define BIAS_IDX(ex) ((((int *)ex)[6])-1)

int labelsize;
int numdatasets;
int dim;
int* D_count;

// comparison function for sorting examples 
int comp(const void *a, const void *b) {
  // sort by extended label first, and whole example second...
  int c = memcmp(*((char **)a) + sizeof(int), 
                 *((char **)b) + sizeof(int), 
                 labelsize*sizeof(int));
  if (c)
    return c;
  
  // labels are the same  
  int alen = **((int **)a);
  int blen = **((int **)b);
  if (alen == blen)
    return memcmp(*((char **)a) + sizeof(int), 
                  *((char **)b) + sizeof(int), 
                  alen);
  return ((alen < blen) ? -1 : 1);
}

// a collapsed example is a sequence of examples
struct collapsed {
  char **seq;
  int num;
};

// set of collapsed examples
struct data {
  collapsed *x;
  int num;
  int numblocks;
  int *blocksizes;
  float *regmult;
  float *learnmult;
};

// seed the random number generator with the current time
void seed_time() {
  struct timeval tp;
  check(gettimeofday(&tp, NULL) == 0);
  srand48((long)tp.tv_usec);
}

static inline double min(double x, double y) { return (x <= y ? x : y); }
static inline double max(double x, double y) { return (x <= y ? y : x); }

// gradient descent
void gd(double C, double J, double lambda, data X, double **w, double ***bias, double **lb) {
  int num = X.num;
  
  // state for random permutations
  int *perm = (int *)malloc(sizeof(int)*X.num);
  check(perm != NULL);

  // state for small cache
  int *W = (int *)malloc(sizeof(int)*num);
  check(W != NULL);
  for (int j = 0; j < num; j++)
    W[j] = 0;

  int t = 0;
  while (t < ITER) {
    // pick random permutation
    for (int i = 0; i < num; i++)
      perm[i] = i;
    for (int swapi = 0; swapi < num; swapi++) {
      int swapj = (int)(drand48()*(num-swapi)) + swapi;
      int tmp = perm[swapi];
      perm[swapi] = perm[swapj];
      perm[swapj] = tmp;
    }

    // count number of examples in the small cache
    int cnum = 0;
    for (int i = 0; i < num; i++) {
      if (W[i] <= INCACHE)
        cnum++;
    }

    for (int swapi = 0; swapi < num; swapi++) {
      // select example
      int i = perm[swapi];
      collapsed x = X.x[i];

      // skip if example is not in small cache
      if (W[i] > INCACHE) {
        W[i]--;
        continue;
      }

      // learning rate
      double T = t + 1000.0;
      double rateX = cnum * C / T;
      double rateR = 1.0 / T;

      if (t % 10000 == 0) {
        printf(".");
        fflush(stdout);
      }
      t++;
      
      // compute max over latent placements
      int M = -1;
      int M_bias = -1;
      double V = 0;
      double V_bias = 0;
      
      for (int m = 0; m < x.num; m++) {
        double val = 0, val_bias = 0;
        char *ptr = x.seq[m];
        int bias_idx = BIAS_IDX(ptr);
        float *data = EX_DATA(ptr);
        int blocks = NUM_NONZERO(ptr);
        for (int j = 0; j < blocks; j++) {
          int b = BLOCK_IDX(data);
          data++;
          for (int k = 0; k < X.blocksizes[b]; k++){
            val += w[b][k] * data[k];
            val_bias += bias[bias_idx][b][k] * data[k];
          }
          data += X.blocksizes[b];
        }
        val_bias += val;
        if (M < 0 || val > V) {
          M = m;
          V = val;
        }
        
        if(M_bias<0 || val_bias > V_bias){
          M_bias = m;
          V_bias = val_bias;
        }
      }
      
      // update model
      for (int j = 0; j < X.numblocks; j++) {
        double mult = rateR * X.regmult[j] * X.learnmult[j];
        for (int k = 0; k < X.blocksizes[j]; k++) {
          double w_j_k = mult * w[j][k];
          w[j][k] -= w_j_k;
          for(int bias_i = 0; bias_i < numdatasets; ++bias_i){
            double bias_i_j_k = mult * bias[bias_i][j][k];
            bias[bias_i][j][k] -= (lambda * bias_i_j_k);
          }
        }
      }
      
      char *ptr = x.seq[M];
      int label = LABEL(ptr);

      if (label * V < 1.0) {
        W[i] = 0;
        float *data = EX_DATA(ptr);
        int blocks = NUM_NONZERO(ptr);
        int bias_idx = BIAS_IDX(ptr);
        for (int j = 0; j < blocks; j++) {
          int b = BLOCK_IDX(data);
          double mult = (label > 0 ? J : -1) * rateX * X.learnmult[b];
          data++;
          for (int k = 0; k < X.blocksizes[b]; k++)
            w[b][k] += mult * data[k];
          data += X.blocksizes[b];
        }
      }
      
      ptr = x.seq[M_bias];
      label = LABEL(ptr);

      if (label * V_bias < 1.0) {
        W[i] = 0;
        float *data = EX_DATA(ptr);
        int blocks = NUM_NONZERO(ptr);
        int bias_idx = BIAS_IDX(ptr);
        for (int j = 0; j < blocks; j++) {
          int b = BLOCK_IDX(data);
          double mult = (label > 0 ? J : -1) * rateX * X.learnmult[b]/D_count[bias_idx];
          double mult_bias = (label > 0 ? J : -1) * rateX * X.learnmult[b]/D_count[bias_idx];
          data++;
          for (int k = 0; k < X.blocksizes[b]; k++){
            w[b][k] += mult * data[k];
            bias[bias_idx][b][k] += mult_bias * data[k];
          }
          data += X.blocksizes[b];
        }
      }
      
      if (label == -1 && label * V_bias >= 1.0 && label * V >= 1.0) {
        if (W[i] == INCACHE)
          W[i] = WAIT;
        else
          W[i]++;
      }
      
    }

    // apply lowerbounds
    for (int j = 0; j < X.numblocks; j++) {
      for (int k = 0; k < X.blocksizes[j]; k++) {
        w[j][k] = max(w[j][k], lb[j][k]);
        for(int bias_i = 0; bias_i < numdatasets; ++bias_i)
          bias[bias_i][j][k] = max(bias[bias_i][j][k], lb[j][k]);
      }
    }

  }

  free(perm);
  free(W);
}

// score examples
double *score(data X, char **examples, int num, double **w) {
  double *s = (double *)malloc(sizeof(double)*num);
  check(s != NULL);
  for (int i = 0; i < num; i++) {
    s[i] = 0.0;
    float *data = EX_DATA(examples[i]);
    int blocks = NUM_NONZERO(examples[i]);
    for (int j = 0; j < blocks; j++) {
      int b = BLOCK_IDX(data);
      data++;
      for (int k = 0; k < X.blocksizes[b]; k++)
        s[i] += w[b][k] * data[k];
      data += X.blocksizes[b];
    }
  }
  return s;  
}

double *score_bias(data X, char **examples, int num, double **w, double ***bias) {
  double *s = (double *)malloc(sizeof(double)*num);
  check(s != NULL);
  for (int i = 0; i < num; i++) {
    s[i] = 0.0;
    float *data = EX_DATA(examples[i]);
    int blocks = NUM_NONZERO(examples[i]);
    int bias_idx = BIAS_IDX(examples[i]);
    for (int j = 0; j < blocks; j++) {
      int b = BLOCK_IDX(data);
      data++;
      for (int k = 0; k < X.blocksizes[b]; k++)
        s[i] += (w[b][k] + bias[bias_idx][b][k]) * data[k];
      data += X.blocksizes[b];
    }
  }
  return s;  
}

// merge examples with identical labels
void collapse(data *X, char **examples, int num) {
  collapsed *x = (collapsed *)malloc(sizeof(collapsed)*num);
  check(x != NULL);
  int i = 0;
  x[0].seq = examples;
  x[0].num = 1;
  for (int j = 1; j < num; j++) {
    if (!memcmp(x[i].seq[0]+sizeof(int), examples[j]+sizeof(int), 
                labelsize*sizeof(int))) {
      x[i].num++;
    } else {
      i++;
      x[i].seq = &(examples[j]);
      x[i].num = 1;
    }
  }
  X->x = x;
  X->num = i+1;  
}

int main(int argc, char **argv) {  
  seed_time();
  int count;
  data X;

  // command line arguments
  check(argc == 10);
  double C = atof(argv[1]);
  double J = atof(argv[2]);
  double lambda = atof(argv[3]);
  char *hdrfile = argv[4];
  char *datfile = argv[5];
  char *modfile = argv[6];
  char *inffile = argv[7];
  char *lobfile = argv[8];
  char *countfile = argv[9];

  // read header file
  FILE *f = fopen(hdrfile, "rb");
  check(f != NULL);
  int header[4];
  count = fread(header, sizeof(int), 4, f);
  check(count == 4);
  int num = header[0];
  labelsize = header[1];
  X.numblocks = header[2];
  numdatasets = header[3];
  X.blocksizes = (int *)malloc(X.numblocks*sizeof(int));
  count = fread(X.blocksizes, sizeof(int), X.numblocks, f);
  check(count == X.numblocks);
  X.regmult = (float *)malloc(sizeof(float)*X.numblocks);
  check(X.regmult != NULL);
  count = fread(X.regmult, sizeof(float), X.numblocks, f);
  check(count == X.numblocks);
  X.learnmult = (float *)malloc(sizeof(float)*X.numblocks);
  check(X.learnmult != NULL);
  count = fread(X.learnmult, sizeof(float), X.numblocks, f);
  check(count == X.numblocks);
  check(num != 0);
  fclose(f);
  printf("%d examples with label size %d and %d blocks\n",
         num, labelsize, X.numblocks);
  printf("block size, regularization multiplier, learning rate multiplier\n");
  dim = 0;
  for (int i = 0; i < X.numblocks; i++) {
    dim += X.blocksizes[i];
    printf("%d, %.2f, %.2f\n", X.blocksizes[i], X.regmult[i], X.learnmult[i]);
  }

  //read count file
  D_count = (int*)malloc(sizeof(int) * numdatasets);
  check(D_count != NULL);
  f = fopen(countfile, "rb");
  check(f != NULL);
  for(int i1=0; i1<numdatasets; ++i1) D_count[i1] = 0;
  fread(D_count, sizeof(int), numdatasets, f);
  for(int i1=0; i1<numdatasets; ++i1) printf("d%d: %d\t", i1, D_count[i1]);
  printf("\n");
  fclose(f);

  // read examples
  f = fopen(datfile, "rb");
  check(f != NULL);
  printf("Reading examples\n");
  char **examples = (char **)malloc(num*sizeof(char *));
  check(examples != NULL);
  for (int i = 0; i < num; i++) {

    // we use an extra byte in the end of each example to mark unique
    // we use an extra int at the start of each example to store the 
    // example's byte length (excluding unique flag and this int)
    int buf[labelsize+2];
    count = fread(buf, sizeof(int), labelsize+2, f);
    check(count == labelsize+2);
    // byte length of an example's data segment
    int len = sizeof(int)*(labelsize+2) + sizeof(float)*buf[labelsize+1];
    
    // memory for data, an initial integer, and a final byte
    examples[i] = (char *)malloc(sizeof(int)+len+1);
    check(examples[i] != NULL);

    // set data segment's byte length
    ((int *)examples[i])[0] = len;

    // set the unique flag to zero
    examples[i][sizeof(int)+len] = 0;

    // copy label data into example
    for (int j = 0; j < labelsize+2; j++)
      ((int *)examples[i])[j+1] = buf[j];

    // read the rest of the data segment into the example
    count = fread(examples[i]+sizeof(int)*(labelsize+3), 1, 
                  len-sizeof(int)*(labelsize+2), f);
    check(count == len-sizeof(int)*(labelsize+2));
  }
  fclose(f);
  printf("done\n");

  // sort
  printf("Sorting examples\n");
  char **sorted = (char **)malloc(num*sizeof(char *));
  check(sorted != NULL);
  memcpy(sorted, examples, num*sizeof(char *));
  qsort(sorted, num, sizeof(char *), comp);
  printf("done\n");

  // find unique examples
  int i = 0;
  int len = *((int *)sorted[0]);
  sorted[0][sizeof(int)+len] = 1;
  for (int j = 1; j < num; j++) {
    int alen = *((int *)sorted[i]);
    int blen = *((int *)sorted[j]);
    if (alen != blen || 
        memcmp(sorted[i] + sizeof(int), sorted[j] + sizeof(int), alen)) {
      i++;
      sorted[i] = sorted[j];
      sorted[i][sizeof(int)+blen] = 1;
    }
  }
  int num_unique = i+1;
  printf("%d unique examples\n", num_unique);

  // collapse examples
  collapse(&X, sorted, num_unique);
  printf("%d collapsed examples\n", X.num);

  // initial model
  double **w = (double **)malloc(sizeof(double *)*X.numblocks);
  check(w != NULL);
  double ***bias;
  
  f = fopen(modfile, "rb");
  for (int i = 0; i < X.numblocks; i++) {
    w[i] = (double *)malloc(sizeof(double)*X.blocksizes[i]);
    check(w[i] != NULL);
    count = fread(w[i], sizeof(double), X.blocksizes[i], f);
    check(count == X.blocksizes[i]);
  }
  
  if(numdatasets > 0){
    bias = (double ***)malloc(sizeof(double **) * numdatasets);
    for(int i=0; i<numdatasets; ++i){
      bias[i] = (double **)malloc(sizeof(double **) * X.numblocks);
      check(bias[i] != NULL);
      for(int j=0; j<X.numblocks; ++j){
        bias[i][j] = (double*) malloc(sizeof(double) * X.blocksizes[j]);
        check(bias[i][j] != NULL);
        count = fread(bias[i][j], sizeof(double), X.blocksizes[j], f);
        check(count == X.blocksizes[j]);
      }
    }
    check(bias != NULL);
  }
  
  fclose(f);

  // lower bounds
  double **lb = (double **)malloc(sizeof(double *)*X.numblocks);
  check(lb != NULL);
  f = fopen(lobfile, "rb");
  for (int i = 0; i < X.numblocks; i++) {
    lb[i] = (double *)malloc(sizeof(double)*X.blocksizes[i]);
    check(lb[i] != NULL);
    count = fread(lb[i], sizeof(double), X.blocksizes[i], f);
    check(count == X.blocksizes[i]);
  }
  fclose(f);
  
  // train
  printf("Training");
  gd(C, J, lambda, X, w, bias, lb);
  printf("done\n");

  // save model
  printf("Saving model\n");
  f = fopen(modfile, "wb");
  check(f != NULL);
  for (int i = 0; i < X.numblocks; i++) {
    count = fwrite(w[i], sizeof(double), X.blocksizes[i], f);
    check(count == X.blocksizes[i]);
  }
  
  for (int i=0; i<numdatasets; i++){
    for(int j=0; j<X.numblocks; ++j){
      count = fwrite(bias[i][j], sizeof(double), X.blocksizes[j], f);
      check(count == X.blocksizes[j]);
    }
  }
  fclose(f);

  // score examples
  printf("Scoring\n");
  double *s = score(X, examples, num, w);
  double *s_bias = score_bias(X, examples, num, w, bias);
  
  // Write info file
  printf("Writing info file\n");
  f = fopen(inffile, "w");
  check(f != NULL);
  for (int i = 0; i < num; i++) {
    int len = ((int *)examples[i])[0];
    int bias_idx = BIAS_IDX(examples[i]);
    // label, score, unique flag, biasidx, score_bias
    count = fprintf(f, "%d\t%f\t%d\t%d\t%f\n", ((int *)examples[i])[1], s[i], 
                    (int)examples[i][sizeof(int)+len], bias_idx, s_bias[i]);
    check(count > 0);
  }
  fclose(f);
  
  printf("Freeing memory\n");
  for (int i = 0; i < X.numblocks; i++) {
    free(w[i]);
    free(lb[i]);
  }
  free(w);
  free(lb);
  free(s);
  for (int i = 0; i < num; i++)
    free(examples[i]);
  free(examples);
  free(sorted);
  free(X.x);
  free(X.blocksizes);
  free(X.regmult);
  free(X.learnmult);

  return 0;
}
