#include "PPintrin.h"

// implementation of absSerial(), but it is vectorized using PP intrinsics
void absVector(float *values, float *output, int N)
{
  __pp_vec_float x;
  __pp_vec_float result;
  __pp_vec_float zero = _pp_vset_float(0.f);
  __pp_mask maskAll, maskIsNegative, maskIsNotNegative;

  //  Note: Take a careful look at this loop indexing.  This example
  //  code is not guaranteed to work when (N % VECTOR_WIDTH) != 0.
  //  Why is that the case?
  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {

    // All ones
    maskAll = _pp_init_ones();

    // All zeros
    maskIsNegative = _pp_init_ones(0);

    // Load vector of values from contiguous memory addresses
    _pp_vload_float(x, values + i, maskAll); // x = values[i];

    // Set mask according to predicate
    _pp_vlt_float(maskIsNegative, x, zero, maskAll); // if (x < 0) {

    // Execute instruction using mask ("if" clause)
    _pp_vsub_float(result, zero, x, maskIsNegative); //   output[i] = -x;

    // Inverse maskIsNegative to generate "else" mask
    maskIsNotNegative = _pp_mask_not(maskIsNegative); // } else {

    // Execute instruction ("else" clause)
    _pp_vload_float(result, values + i, maskIsNotNegative); //   output[i] = x; }

    // Write results back to memory
    _pp_vstore_float(output + i, result, maskAll);
  }
}

void clampedExpVector(float *values, int *exponents, float *output, int N)
{
  //
  // PP STUDENTS TODO: Implement your vectorized version of
  // clampedExpSerial() here.
  //
  // Your solution should work for any value of
  // N and VECTOR_WIDTH, not just when VECTOR_WIDTH divides N
  //
  __pp_vec_float x, x_tmp = _pp_vset_float(0.f);
  __pp_vec_int y, y_tmp = _pp_vset_int(1);
  __pp_vec_float result;
  __pp_vec_float clamp_th = _pp_vset_float(9.999999f);
  __pp_vec_int zero = _pp_vset_int(0.f);
  __pp_vec_int one = _pp_vset_int(1);
  __pp_mask maskAll, maskZero, maskNonZero, maskTh;
  int mod = N % VECTOR_WIDTH;
  
  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {
    maskAll = _pp_init_ones();

    // if (N % VECTOR_WIDTH) != 0
    if((i + VECTOR_WIDTH) > N){
      for(int offset = 0; offset < mod; offset++){
        x_tmp.value[offset] = values[i + offset];
        y_tmp.value[offset] = exponents[i + offset];
      }
      _pp_vmove_float(x, x_tmp, maskAll); // x = values[i]
      _pp_vmove_int(y, y_tmp, maskAll); // y = exponents[i]
    }
    else{
      _pp_vload_float(x, values + i, maskAll); // x = values[i]
      _pp_vload_int(y, exponents + i, maskAll); // y = exponents[i]
    }

      _pp_vmove_float(result, x, maskAll); // result = x;

      // if there are exponents equal to 0
      _pp_veq_int(maskZero, y, zero, maskAll); // if (y == 0)
      _pp_vset_float(result, 1.f, maskZero); // output[i] = 1.f

      // if there is non zero exponent (no negative exponent in test data)
      _pp_vsub_int(y, y, one, maskAll); // count --
      _pp_vgt_int(maskNonZero, y, zero, maskAll); // find the non zero exponent
      while(_pp_cntbits(maskNonZero)){
        _pp_vmult_float(result, x, result, maskNonZero); // result *= x
        _pp_vsub_int(y, y, one, maskNonZero); // count --
        _pp_vgt_int(maskNonZero, y, zero, maskAll); // find the non zero exponent
      }
      // clamp the value in result
      _pp_vgt_float(maskTh, result, clamp_th, maskAll); // if result > 9.999999f
      _pp_vset_float(result, 9.999999f, maskTh);
      _pp_vstore_float(output + i, result, maskAll); // output[i] = result
  }
}

// returns the sum of all elements in values
// You can assume N is a multiple of VECTOR_WIDTH
// You can assume VECTOR_WIDTH is a power of 2
float arraySumVector(float *values, int N)
{

  //
  // PP STUDENTS TODO: Implement your vectorized version of arraySumSerial here
  //

  float sum = 0.f;
  __pp_vec_float sumVec = _pp_vset_float(0.f), loadVec;
  __pp_mask maskAll;

  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {
    maskAll = _pp_init_ones();
    _pp_vload_float(loadVec, values + i, maskAll);
    _pp_vadd_float(sumVec, sumVec, loadVec, maskAll);
  }

  for(int j = 0; j < (log(VECTOR_WIDTH) / log(2)); j++){
    _pp_hadd_float(sumVec, sumVec);
    _pp_interleave_float(sumVec, sumVec);
  }
  sum = sumVec.value[0];
  return sum;
}