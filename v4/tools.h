/***************************************************************************
 * Copyright (2012)2 (03-2014)3 Intel Corporation All Rights Reserved.
 *
 * The source code contained or described herein and all documents related to 
 * the source code ("Material") are owned by Intel Corporation or its suppliers 
 * or licensors. Title to the Material remains with Intel Corporation or its 
 * suppliers and licensors. The Material contains trade secrets and proprietary 
 * and confidential information of Intel or its suppliers and licensors. The 
 * Material is protected by worldwide copyright and trade secret laws and 
 * treaty provisions. No part of the Material may be used, copied, reproduced, 
 * modified, published, uploaded, posted, transmitted, distributed, or disclosed 
 * in any way without Intel’s prior express written permission.
 *
 * No license under any patent, copyright, trade secret or other intellectual 
 * property right is granted to or conferred upon you by disclosure or delivery 
 * of the Materials, either expressly, by implication, inducement, estoppel or 
 * otherwise. Any license under such intellectual property rights must be express 
 * and approved by Intel in writing.
 * ***************************************************************************/

/*****************************************************************************
 * ! Content:
 * ! Implementation example of ISO-3DFD implementation for 
 * !   Intel(R) Xeon Phi(TM) and Intel(R) Xeon.
 * ! Version 00
 * ! leonardo.borges@intel.com
 * ! cedric.andreolli@intel.com
 * !****************************************************************************/

#ifndef _TOOLS_INCLUDE
#define _TOOLS_INCLUDE

#include <stddef.h>
#include <sys/time.h>
#include <time.h>

// NOTE: the use of clock_gettime() below requires you to link
// with -lrt (the real-time clock)
double walltime() // seconds
{
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);

  return ((double)(ts.tv_sec) +
          1e-09 * (double)(ts.tv_nsec));
}

//#if defined(VERIFY_RESULTS)
#include <math.h>
void init_data(float *data, const int dimx, const int dimy, const int dimz)
{
  for (int iz = 0; iz < dimz; iz++)
    for (int iy = 0; iy < dimy; iy++)
      for (int ix = 0; ix < dimx; ix++)
      {
        *data = (float)iz;
        ++data;
      }
}

// naive and slow implementation
void reference_implementation(float *next, float *prev, float *coeff,
                              float *vel,
                              const int n1, const int n2, const int n3, const int half_length)
{
  int n1n2 = n1 * n2;

  for (int iz = 0; iz < n3; iz++)
  {
    for (int iy = 0; iy < n2; iy++)
    {
      for (int ix = 0; ix < n1; ix++)
      {
        if (ix >= half_length && ix < (n1 - half_length) && iy >= half_length && iy < (n2 - half_length) && iz >= half_length && iz < (n3 - half_length))
        {
          float res = prev[iz * n1n2 + iy * n1 + ix] * coeff[0];
          for (int ir = 1; ir <= half_length; ir++)
          {
            res += coeff[ir] * (prev[iz * n1n2 + iy * n1 + ix + ir] + prev[iz * n1n2 + iy * n1 + ix - ir]);                     // horizontal
            res += coeff[ir] * (prev[iz * n1n2 + iy * n1 + ix + ir * n1] + prev[iz * n1n2 + iy * n1 + ix - ir * n1]);           // vertical
            res += coeff[ir] * (prev[iz * n1n2 + iy * n1 + ix + ir * n1 * n2] + prev[iz * n1n2 + iy * n1 + ix - ir * n1 * n2]); // in front / behind
          }
          next[iz * n1n2 + iy * n1 + ix] = 2.0f * prev[iz * n1n2 + iy * n1 + ix] - next[iz * n1n2 + iy * n1 + ix] + res * vel[iz * n1n2 + iy * n1 + ix];
          //
          //printf("(%d %d %d):prev:%.3f next:%.3f vel:%.3f\n",)
        }
      }
    }
  }
}

void reference_implementation_mpi(float *next, float *prev, float *coeff, float *vel, const int n1, const int n2, const int n3, const int half_length, const int blockSize)
{
  int n1n2 = n1 * n2;

  for (int iz = half_length; iz < half_length + blockSize; iz++)
  {
    for (int iy = 0; iy < n2; iy++)
    {
      for (int ix = 0; ix < n1; ix++)
      {
        if (ix >= half_length && ix < (n1 - half_length) && iy >= half_length && iy < (n2 - half_length))
        {
          float res = prev[iz * n1n2 + iy * n1 + ix] * coeff[0];
          for (int ir = 1; ir <= half_length; ir++)
          {
            res += coeff[ir] * (prev[iz * n1n2 + iy * n1 + ix + ir] + prev[iz * n1n2 + iy * n1 + ix - ir]);                     // horizontal
            res += coeff[ir] * (prev[iz * n1n2 + iy * n1 + ix + ir * n1] + prev[iz * n1n2 + iy * n1 + ix - ir * n1]);           // vertical
            res += coeff[ir] * (prev[iz * n1n2 + iy * n1 + ix + ir * n1 * n2] + prev[iz * n1n2 + iy * n1 + ix - ir * n1 * n2]); // in front / behind
          }
          next[iz * n1n2 + iy * n1 + ix] = 2.0f * prev[iz * n1n2 + iy * n1 + ix] - next[iz * n1n2 + iy * n1 + ix] + res * vel[iz * n1n2 + iy * n1 + ix];
          //printf("(%d %d %d):prev:%.3f next:%.3f vel:%.3f\n",)
        }
      }
    }
  }
}

void reference_implementation_mpi_x_y(float *next, float *prev, float *coeff, float *vel, const int n1, const int n2, const int n3, const int half_length, const int blockSize)
{
  int n1n2 = n1 * n2;

  for (int iz = 0; iz < n3; iz++)
  {
    for (int iy = HALF_LENGTH; iy < HALF_LENGTH + n2; iy++)
    {
      for (int ix = HALF_LENGTH; ix < HALF_LENGTH + n1; ix++)
      {
        if (iz >= half_length && iz < (n3 - half_length))
        {
          float res = prev[iz * n1n2 + iy * n1 + ix] * coeff[0];
          for (int ir = 1; ir <= half_length; ir++)
          {
            res += coeff[ir] * (prev[iz * n1n2 + iy * n1 + ix + ir] + prev[iz * n1n2 + iy * n1 + ix - ir]);                     // horizontal
            res += coeff[ir] * (prev[iz * n1n2 + iy * n1 + ix + ir * n1] + prev[iz * n1n2 + iy * n1 + ix - ir * n1]);           // vertical
            res += coeff[ir] * (prev[iz * n1n2 + iy * n1 + ix + ir * n1 * n2] + prev[iz * n1n2 + iy * n1 + ix - ir * n1 * n2]); // in front / behind
          }
          next[iz * n1n2 + iy * n1 + ix] = 2.0f * prev[iz * n1n2 + iy * n1 + ix] - next[iz * n1n2 + iy * n1 + ix] + res * vel[iz * n1n2 + iy * n1 + ix];
          //printf("(%d %d %d):prev:%.3f next:%.3f vel:%.3f\n",)
        }
      }
    }
  }
}

void reference_implementation_mpi_2D(float *next, float *prev, float *coeff, float *vel, float *preHalo,const int n3, const int half_length, const int xDivisionSize, const int yDivisionSize)
{
  int n2n3 = (2*half_length+yDivisionSize) * n3;

  for (int ix = half_length; ix < half_length + xDivisionSize; ix++)
  {
    for (int iy = half_length; iy < half_length + yDivisionSize; iy++)
    {
      for (int iz = 0; iz < n3; iz++)
      {
        if (iz >= half_length && iz < (n3 - half_length))
        {
          float res = prev[ix * n2n3 + iy * n3 + iz] * coeff[0];
          for (int ir = 1; ir <= half_length; ir++)
          {
            res += coeff[ir] * (prev[ix * n2n3 + iy * n3 + iz + ir] + prev[ix * n2n3 + iy * n3 + iz - ir]);           // horizontal
            res += coeff[ir] * (prev[ix * n2n3 + iy * n3 + iz + ir * n3] + prev[ix * n2n3 + iy * n3 + iz - ir * n3]); // vertical
            float prevLeft = prev[ix * n2n3 + iy * n3 + iz + ir * n2n3];
            float prevRight = prev[ix * n2n3 + iy * n3 + iz - ir * n2n3];
            if (ix < 2 * half_length)
            {
              prevLeft = preHalo[(ix - half_length) * n2n3 + iy * n3 + iz];
            }
            if (ix > xDivisionSize)
            {
              prevRight = preHalo[(ix - xDivisionSize + half_length) * n2n3 + iy * n3 + iz];
            }
            res += coeff[ir] * (prevLeft + prevRight); // in front / behind
          }
          next[ix * n2n3 + iy * n3 + iz] = 2.0f * prev[ix * n2n3 + iy * n3 + iz] - next[ix * n2n3 + iy * n3 + iz] + res * vel[ix * n2n3 + iy * n3 + iz];
          //printf("(%d %d %d):prev:%.3f next:%.3f vel:%.3f\n",ix,iy,iz,prev[ix*n2n3+iy*n3+iz],next[ix*n2n3+iy*n3+iz],vel[ix*n2n3+iy*n3+iz]);
        }
      }
    }
  }
}

void copy_next_to_send(float *next, float *send, const int half_length, const int xDivisionSize,const int yDivisionSize, const int n3)
{

  int n2n3 = (2 * half_length + yDivisionSize) * n3;

  int key=-1;
  int ix;
  for (int iy = half_length; iy < half_length + yDivisionSize; iy++)
  {
    for (int iz = 0; iz < n3; iz++)
    {
      for (ix = half_length; ix < 2 * half_length; ix++)
      {
	key=(ix - half_length) * n2n3 + iy * n3 + iz;
        send[(ix - half_length) * n2n3 + iy * n3 + iz] = next[ix * n2n3 + iy * n3 + iz];
      }
	//printf("copy_next_to_send(%d,%d,%d):%.3f\n",ix,iy,iz,send[key]);
      for (ix = xDivisionSize;ix<xDivisionSize+half_length;ix++){
	key=(ix - xDivisionSize + half_length) * n2n3 + iy * n3 + iz;
        send[(ix - xDivisionSize + half_length) * n2n3 + iy * n3 + iz] = next[ix * n2n3 + iy * n3 + iz];
      }
	//printf("copy_next_to_send(%d,%d,%d):%.3f\n",ix,iy,iz,send[key]);
    }
  }
}

bool within_epsilon(float *output, float *reference, const int dimx, const int dimy, const int dimz, const int radius, const int zadjust = 0, const float delta = 0.0001f)
{
  bool retval = true;
  float abs_delta = fabsf(delta);
  for (int iz = 0; iz < dimz; iz++)
  {
    for (int iy = 0; iy < dimy; iy++)
    {
      for (int ix = 0; ix < dimx; ix++)
      {
        if (ix >= radius && ix < (dimx - radius) && iy >= radius && iy < (dimy - radius) && iz >= radius && iz < (dimz - radius + zadjust))
        {
          float difference = fabsf(*reference - *output);
          if (difference > delta)
          {
            retval = false;
            printf(" ERROR: (%d,%d,%d)\t%.2f instead of %.2f\n", ix, iy, iz, *output, *reference);
            return false;
          }
        }
        ++output;
        ++reference;
      }
    }
  }
  return retval;
}

//#endif /* VERIFY_RESULTS */

#endif /*_TOOLS_INCLUDE */
