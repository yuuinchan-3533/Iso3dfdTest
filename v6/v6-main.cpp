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
! Content:
! Implementation example of ISO-3DFD implementation for 
!   Intel(R) Xeon Phi(TM) and Intel(R) Xeon.
! Version 00
! leonardo.borges@intel.com
! cedric.andreolli@intel.com
!****************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <mpi.h>
#include <math.h>

#include "iso-3dfd.h"
#include "tools.h"
int rank, pSize;						   //rank：当前进程ID，pSize：总的进程数
int xProcessNum, yProcessNum, zProcessNum; //x轴上划分的进程数、y轴上划分的进程数、z轴上划分的进程数
int xBlockSize, yBlockSize, zBlockSize;	//x轴每个进程算的格点数,y轴每个进程算得格点数、z轴每个进程算得格点数

typedef struct
{
	size_t n1; // First dimension
	size_t n2; // Second dimension
	size_t n3; // Third dimension
	int num_threads;
	int nreps;		  // number of time-steps, over which performance is averaged
	size_t n1_Tblock; // Thread blocking on 1st dimension
	size_t n2_Tblock; // Thread blocking on 2nd dimension
	size_t n3_Tblock; // Thread blocking on 3rd dimension
	float *prev;
	float *next;
	float *vel;

	float *leftSendBlock;  // x dimension x-
	float *rightSendBlock; // x dimension x+
	float *frontSendBlock; // y dimension y-
	float *backSendBlock;  // y dimension y+
	float *upSendBlock;	// z dimension z-
	float *downSendBlock;  // z dimension z+

	float *leftRecvBlock;  // x dimension x-
	float *rightRecvBlock; // x dimension x+
	float *frontRecvBlock; // y dimension y-
	float *backRecvBlock;  // y dimension y+
	float *upRecvBlock;	// z dimension z-
	float *downRecvBlock;  // z dimension z+
} Parameters;

//Function used for initialization

void initiate_v6(int rank, float *ptr_prev, float *ptr_next, float *ptr_vel, Parameters *p, int xDivisionSize, int yDivisionSize, int zDivisionSize,const int xOffSet, const int yOffSet, const int zOffSet)
{

	int n1 = 2 * HALF_LENGTH + xDivisionSize;
	int n2 = 2 * HALF_LENGTH + yDivisionSize;
	int n3 = 2 * HALF_LENGTH + zDivisionSize;
	//	printf("%s %d\n",__FILE__,__LINE__);

	for (int iz = 0; iz < n3; iz++)
	{
		for (int iy = 0; iy < n2; iy++)
		{
			for (int ix = 0; ix < n1; ix++)
			{
				int key = iz * n1 * n2 + iy * n1 + ix; //[z][y][x]
				ptr_prev[key] = sin((ix + xOffSet) * 1 + (iy + yOffSet) * 10 + (iz + zOffSet) * 100);
				ptr_next[key] = cos((ix + xOffSet) * 1 + (iy + yOffSet) * 10 + (iz + zOffSet) * 100);
				ptr_vel[key] = 2250000.0f * DT * DT;
				//printf("initiate(%d %d %d):%.3f %.3f\n",ix+xOffSet,iy+yOffSet,iz,ptr_prev[key],ptr_next[key]);
			}
		}
	}
}

void copy_data_to_left_send_block(Parameters *p, int xDivisionSize, int yDivisionSize, int zDivisionSize)
{
	int n1 = 2 * HALF_LENGTH + xDivisionSize;
	int n2 = 2 * HALF_LENGTH + yDivisionSize;
	int n3 = 2 * HALF_LENGTH + zDivisionSize;
	int ix, iy, iz;
	int sendBlockKey = 0;
	for (int iz = 0; iz < n3; iz++)
	{
		for (int iy = 0; iy < n2; iy++)
		{
			for (int ix = 0; ix < HALF_LENGTH; ix++)
			{

				p->leftSendBlock[sendBlockKey++] = p->prev[iz * n2 * n1 + iy * n1 + ix + HALF_LENGTH];
				//printf("initiate(%d %d %d):%.3f %.3f\n",ix+xOffSet,iy+yOffSet,iz,ptr_prev[key],ptr_next[key]);
			}
		}
	}
}

void copy_data_to_left_halo(Parameters *p, int xDivisionSize, int yDivisionSize, int zDivisionSize)
{
	int n1 = 2 * HALF_LENGTH + xDivisionSize;
	int n2 = 2 * HALF_LENGTH + yDivisionSize;
	int n3 = 2 * HALF_LENGTH + zDivisionSize;
	int sendBlockKey = 0;
	int ix, iy, iz;

	for (int iz = 0; iz < n3; iz++)
	{
		for (int iy = 0; iy < n2; iy++)
		{
			for (int ix = 0; ix < HALF_LENGTH; ix++)
			{

				p->prev[iz * n2 * n1 + iy * n1 + ix] = p->leftRecvBlock[sendBlockKey++];
				//printf("initiate(%d %d %d):%.3f %.3f\n",ix+xOffSet,iy+yOffSet,iz,ptr_prev[key],ptr_next[key]);
			}
		}
	}
}

void copy_data_to_right_send_block(Parameters *p, int xDivisionSize, int yDivisionSize, int zDivisionSize)
{
	int n1 = 2 * HALF_LENGTH + xDivisionSize;
	int n2 = 2 * HALF_LENGTH + yDivisionSize;
	int n3 = 2 * HALF_LENGTH + zDivisionSize;
	int sendBlockKey = 0;
	int ix, iy, iz;
	//leftBlock[HALF][y][z]

	for (int iz = 0; iz < n3; iz++)
	{
		for (int iy = 0; iy < n2; iy++)
		{
			for (int ix = 0; ix < HALF_LENGTH; ix++)
			{

				p->rightSendBlock[sendBlockKey++] = p->prev[iz * n2 * n1 + iy * n1 + ix + xDivisionSize];
				//printf("initiate(%d %d %d):%.3f %.3f\n",ix+xOffSet,iy+yOffSet,iz,ptr_prev[key],ptr_next[key]);
			}
		}
	}
}

void copy_data_to_right_halo(Parameters *p, int xDivisionSize, int yDivisionSize, int zDivisionSize)
{
	int n1 = 2 * HALF_LENGTH + xDivisionSize;
	int n2 = 2 * HALF_LENGTH + yDivisionSize;
	int n3 = 2 * HALF_LENGTH + zDivisionSize;
	int sendBlockKey = 0;
	int ix, iy, iz;
	for (int iz = 0; iz < n3; iz++)
	{
		for (int iy = 0; iy < n2; iy++)
		{
			for (int ix = 0; ix < HALF_LENGTH; ix++)
			{

				p->prev[iz * n2 * n1 + iy * n1 + (ix + xDivisionSize + HALF_LENGTH)] = p->prev[sendBlockKey++];
				//printf("initiate(%d %d %d):%.3f %.3f\n",ix+xOffSet,iy+yOffSet,iz,ptr_prev[key],ptr_next[key]);
			}
		}
	}
}

void copy_data_to_front_send_block(Parameters *p, int xDivisionSize, int yDivisionSize, int zDivisionSize)
{
	int n1 = 2 * HALF_LENGTH + xDivisionSize;
	int n2 = 2 * HALF_LENGTH + yDivisionSize;
	int n3 = 2 * HALF_LENGTH + zDivisionSize;
	int ix, iy, iz;
	int sendBlockKey = 0;
	for (int iz = 0; iz < n3; iz++)
	{
		for (int iy = 0; iy < HALF_LENGTH; iy++)
		{
			for (int ix = 0; ix < n1; ix++)
			{

				p->frontSendBlock[sendBlockKey++] = p->prev[iz * n2 * n1 + (iy + HALF_LENGTH) * n1 + ix];
				//printf("initiate(%d %d %d):%.3f %.3f\n",ix+xOffSet,iy+yOffSet,iz,ptr_prev[key],ptr_next[key]);
			}
		}
	}
}
void copy_data_to_front_halo(Parameters *p, int xDivisionSize, int yDivisionSize, int zDivisionSize)
{
	int n1 = 2 * HALF_LENGTH + xDivisionSize;
	int n2 = 2 * HALF_LENGTH + yDivisionSize;
	int n3 = 2 * HALF_LENGTH + zDivisionSize;
	int ix, iy, iz;
	int sendBlockKey = 0;

	for (int iz = 0; iz < n3; iz++)
	{
		for (int iy = 0; iy < HALF_LENGTH; iy++)
		{
			for (int ix = 0; ix < n1; ix++)
			{
				p->prev[iz * n2 * n1 + iy * n1 + ix] = p->frontRecvBlock[sendBlockKey++];
			}
		}
	}
}

void copy_data_to_back_send_block(Parameters *p, int xDivisionSize, int yDivisionSize, int zDivisionSize)
{
	int n1 = 2 * HALF_LENGTH + xDivisionSize;
	int n2 = 2 * HALF_LENGTH + yDivisionSize;
	int n3 = 2 * HALF_LENGTH + zDivisionSize;
	int ix, iy, iz;
	int sendBlockKey = 0;

	for (int iz = 0; iz < n3; iz++)
	{
		for (int iy = 0; iy < HALF_LENGTH; iy++)
		{
			for (int ix = 0; ix < n1; ix++)
			{

				p->backSendBlock[sendBlockKey++] = p->prev[iz * n2 * n1 + (iy + yDivisionSize) * n1 + ix];
				//printf("initiate(%d %d %d):%.3f %.3f\n",ix+xOffSet,iy+yOffSet,iz,ptr_prev[key],ptr_next[key]);
			}
		}
	}
}

void copy_data_to_back_halo(Parameters *p, int xDivisionSize, int yDivisionSize, int zDivisionSize)
{
	int n1 = 2 * HALF_LENGTH + xDivisionSize;
	int n2 = 2 * HALF_LENGTH + yDivisionSize;
	int n3 = 2 * HALF_LENGTH + zDivisionSize;
	int ix, iy, iz;
	int sendBlockKey = 0;

	for (int iz = 0; iz < n3; iz++)
	{
		for (int iy = 0; iy < HALF_LENGTH; iy++)
		{
			for (int ix = 0; ix < n1; ix++)
			{
				p->prev[iz * n2 * n1 + (iy + yDivisionSize + HALF_LENGTH) * n1 + ix] = p->backRecvBlock[sendBlockKey++];
			}
		}
	}
}
void copy_data_to_up_send_block(Parameters *p, int xDivisionSize, int yDivisionSize, int zDivisionSize)
{
	int n1 = 2 * HALF_LENGTH + xDivisionSize;
	int n2 = 2 * HALF_LENGTH + yDivisionSize;
	int n3 = 2 * HALF_LENGTH + zDivisionSize;
	int ix, iy, iz;
	int sendBlockKey = 0;

	for (int iz = 0; iz < HALF_LENGTH; iz++)
	{
		for (int iy = 0; iy < n2; iy++)
		{
			for (int ix = 0; ix < n1; ix++)
			{
				p->upSendBlock[sendBlockKey++] = p->prev[(iz + HALF_LENGTH) * n2 * n1 + iy * n1 + ix];
				//printf("initiate(%d %d %d):%.3f %.3f\n",ix+xOffSet,iy+yOffSet,iz,ptr_prev[key],ptr_next[key]);
			}
		}
	}
}

void copy_data_to_up_halo(Parameters *p, int xDivisionSize, int yDivisionSize, int zDivisionSize)
{
	int n1 = 2 * HALF_LENGTH + xDivisionSize;
	int n2 = 2 * HALF_LENGTH + yDivisionSize;
	int n3 = 2 * HALF_LENGTH + zDivisionSize;
	int ix, iy, iz;
	int sendBlockKey = 0;

	for (int iz = 0; iz < HALF_LENGTH; iz++)
	{
		for (int iy = 0; iy < n2; iy++)
		{
			for (int ix = 0; ix < n1; ix++)
			{
				p->prev[iz * n2 * n1 + iy * n1 + ix] = p->upRecvBlock[sendBlockKey++];
			}
		}
	}
}

void copy_data_to_down_send_block(Parameters *p, int xDivisionSize, int yDivisionSize, int zDivisionSize)
{
	int n1 = 2 * HALF_LENGTH + xDivisionSize;
	int n2 = 2 * HALF_LENGTH + yDivisionSize;
	int n3 = 2 * HALF_LENGTH + zDivisionSize;
	int ix, iy, iz;
	int sendBlockKey = 0;

	for (int iz = 0; iz < HALF_LENGTH; iz++)
	{
		for (int iy = 0; iy < n2; iy++)
		{
			for (int ix = 0; ix < n1; ix++)
			{
				p->downSendBlock[sendBlockKey++] = p->prev[(iz + zDivisionSize) * n2 * n1 + iy * n1 + ix];
			}
		}
	}
}

void copy_data_to_down_halo(Parameters *p, int xDivisionSize, int yDivisionSize, int zDivisionSize)
{
	int n1 = 2 * HALF_LENGTH + xDivisionSize;
	int n2 = 2 * HALF_LENGTH + yDivisionSize;
	int n3 = 2 * HALF_LENGTH + zDivisionSize;
	int ix, iy, iz;
	int sendBlockKey = 0;

	for (int iz = 0; iz < HALF_LENGTH; iz++)
	{
		for (int iy = 0; iy < n2; iy++)
		{
			for (int ix = 0; ix < n1; ix++)
			{
				p->prev[(iz + zDivisionSize + HALF_LENGTH) * n2 * n1 + iy * n1 + ix] = p->upRecvBlock[sendBlockKey++];
			}
		}
	}
}

void copy_data_to_send_block(Parameters *p, int xDivisionSize, int yDivisionSize, int zDivisionSize)
{
	copy_data_to_left_send_block(p, xDivisionSize, yDivisionSize, zDivisionSize);  //x Dimension
	copy_data_to_right_send_block(p, xDivisionSize, yDivisionSize, zDivisionSize); // x Dimension
	copy_data_to_front_send_block(p, xDivisionSize, yDivisionSize, zDivisionSize); //y Dimension
	copy_data_to_back_send_block(p, xDivisionSize, yDivisionSize, zDivisionSize);  //y Dimension
	copy_data_to_up_send_block(p, xDivisionSize, yDivisionSize, zDivisionSize);	//z Dimension
	copy_data_to_down_send_block(p, xDivisionSize, yDivisionSize, zDivisionSize);  //z Dimension
}
void update_halo(Parameters *p, int xDivisionSize, int yDivisionSize, int zDivisioinSize, const int left, const int right, const int front, const int back, const int up, const int down)
{
	if (left != -1)
	{
		copy_data_to_left_halo(p, xDivisionSize, yDivisionSize, zDivisioinSize);
	}
	if (right != -1)
	{
		copy_data_to_right_halo(p, xDivisionSize, yDivisionSize, zDivisioinSize);
	}
	if (front != -1)
	{
		copy_data_to_front_halo(p, xDivisionSize, yDivisionSize, zDivisioinSize);
	}
	if (back != -1)
	{
		copy_data_to_back_halo(p, xDivisionSize, yDivisionSize, zDivisioinSize);
	}
	if (up != -1)
	{
		copy_data_to_up_halo(p, xDivisionSize, yDivisionSize, zDivisioinSize);
	}
	if (down != -1)
	{
		copy_data_to_down_halo(p, xDivisionSize, yDivisionSize, zDivisioinSize);
	}
}

void output_v6(Parameters *p, int rank, const int xDivisionSize, const int yDivisionSize, const int zDivisionSize, const int xOffSet, const int yOffSet, const int zOffSet)
{
	int n1 = 2 * HALF_LENGTH + xDivisionSize;
	int n2 = 2 * HALF_LENGTH + yDivisionSize;
	int n3 = 2 * HALF_LENGTH + zDivisionSize;
	for (int rk = 0; rk < pSize; rk++)
	{
		fflush(stdout);
		MPI_Barrier(MPI_COMM_WORLD);
		if (rk == rank)
		{
			for (int ix = HALF_LENGTH; ix < n1 - HALF_LENGTH; ix++)
			{
				for (int iy = HALF_LENGTH; iy < n2 - HALF_LENGTH; iy++)
				{
					for (int iz = HALF_LENGTH; iz < n3 - HALF_LENGTH; iz++)
					{
						int key = iz * n1 * n2 + iy * n1 + ix; //[z][y][x]
						printf("rank:%d(%d %d %d)%.3f\n", rank, ix + xOffSet, iy + yOffSet, iz + zOffSet, p->prev[key]);
						//printf("initiate(%d %d %d):%.3f %.3f\n",ix+xOffSet,iy+yOffSet,iz,ptr_prev[key],ptr_next[key]);
					}
				}
			}
		}
	}
}

void initiate_params_v6(int n1, int n2, int n3)
{
	int x, y, z;
	float diff = n1 > n2 ? (n1 + 0.0) / (n2 + 0.0) : (n2 + 0.0) / (n1 + 0.0);
	float minn = 100000.0;
	float tempdiff = 0;
	x = floor(pow(pSize * n1 * n1 * 1.0 / (n2 * n3), 1.0 / 3));

	if ((n2 + 0.0) / n1 >= 1)
	{
		y = floor((x + 0.0) * n2 / n1);
	}
	else
	{
		y = ceil((x + 0.0) * n2 / n1);
	}
	
	z = floor(pSize / (x * y * 1.0));
	if (x * y * z > pSize)
	{
		y = y - 1;
	}
	xProcessNum = x;
	yProcessNum = y;
	zProcessNum = z;

	xBlockSize = ceil((n1 - 2 * HALF_LENGTH) / (xProcessNum + 0.0));
	yBlockSize = ceil((n2 - 2 * HALF_LENGTH) / (yProcessNum + 0.0));
	zBlockSize = ceil((n3 - 2 * HALF_LENGTH) / (zProcessNum + 0.0));
	return;
}

int main(int argc, char **argv)
{
	// Defaults
	MPI_Init(&argc, &argv);				   //MPI初始化语句
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);  //获取当前进程的pID
	MPI_Comm_size(MPI_COMM_WORLD, &pSize); //获取进程总数
	MPI_Status status;
	Parameters p;
	p.n1 = 256; // First dimension x
	p.n2 = 300; // Second dimension y
	p.n3 = 300; // Third dimension z
	p.num_threads = 24;
	p.nreps = 100;									 // number of time-steps, over which performance is averaged
	p.n1_Tblock;									 // Thread blocking on 1st dimension
	p.n2_Tblock;									 // Thread blocking on 2nd dimension
	p.n3_Tblock;									 // Thread blocking on 3rd dimension
#define N2_TBLOCK 1									 // Default thread blocking on 2nd dimension: 1
#define N3_TBLOCK 124								 // Default thread blocking on 3rd dimension: 124
	int left, right, front, back, up, down;			 // 相邻进程编号
	int xDivisionSize, yDivisionSize, zDivisionSize; //该进程在x轴上计算的空间大小、该进程在y轴上计算的空间大小

	//	initiate_params(p.n1, p.n2);
	if ((argc > 1) && (argc < 4))
	{
		printf(" usage: [n1 n2 n3] [# threads] [# iterations] [thread block n1] [thread block n2] [thread block n3]\n");
		exit(1);
	}
	// [n1 n2 n3]
	if (argc >= 4)
	{
		p.n1 = atoi(argv[1]);
		p.n2 = atoi(argv[2]);
		p.n3 = atoi(argv[3]);
	}
	//  [# threads]
	if (argc >= 5)
		p.num_threads = atoi(argv[4]);
	//  [# iterations]
	if (argc >= 6)
		p.nreps = atoi(argv[5]);
	//  [thread block n1] [thread block n2] [thread block n3]
	if (argc >= 7)
	{
		p.n1_Tblock = atoi(argv[6]);
	}
	else
	{
		p.n1_Tblock = p.n1; // Default: no blocking on 1st dim
	}
	if (argc >= 8)
	{
		p.n2_Tblock = atoi(argv[7]);
	}
	else
	{
		p.n2_Tblock = N2_TBLOCK;
	}
	if (argc >= 9)
	{
		p.n3_Tblock = atoi(argv[8]);
	}
	else
	{
		p.n3_Tblock = N3_TBLOCK;
	}

	// Make sure n1 and n1_Tblock are multiple of 16 (to support 64B alignment)
	if ((p.n1 % 16) != 0)
	{
		printf("Parameter n1=%d must be a multiple of 16\n", p.n1);
		exit(1);
	}
	if ((p.n1_Tblock % 16) != 0)
	{
		printf("Parameter n1_Tblock=%d must be a multiple of 16\n", p.n1_Tblock);
		exit(1);
	}

	initiate_params_v6(p.n1, p.n2, p.n3);
	// Make sure nreps is rouded up to next even number (to support swap)
	p.nreps = ((p.nreps + 1) / 2) * 2;

	//printf("n1=%d n2=%d n3=%d nreps=%d num_threads=%d HALF_LENGTH=%d\n", p.n1, p.n2, p.n3, p.nreps, p.num_threads, HALF_LENGTH);
	//printf("n1_thrd_block=%d n2_thrd_block=%d n3_thrd_block=%d\n", p.n1_Tblock, p.n2_Tblock, p.n3_Tblock);

#if (HALF_LENGTH == 4)
	float coeff[HALF_LENGTH + 1] = {
		-2.847222222,
		+1.6,
		-0.2,
		+2.53968e-2,

		-1.785714e-3};
#elif (HALF_LENGTH == 8)
	float coeff[HALF_LENGTH + 1] = {
		-3.0548446,
		+1.7777778,
		-3.1111111e-1,
		+7.572087e-2,
		-1.76767677e-2,
		+3.480962e-3,
		-5.180005e-4,
		+5.074287e-5,
		-2.42812e-6};
#else
#error "HALF_LENGTH not implemented"
#endif
	//Apply the DX DY and DZ to coefficients
	coeff[0] = (3.0f * coeff[0]) / (DXYZ * DXYZ);
	for (int i = 1; i <= HALF_LENGTH; i++)
	{
		coeff[i] = coeff[i] / (DXYZ * DXYZ);
	}

	// Data Arrays
	p.prev = NULL, p.next = NULL, p.vel = NULL;

	// variables for measuring performance
	double wstart, wstop;
	float elapsed_time = 0.0f, throughput_mpoints = 0.0f, mflops = 0.0f;

	left = rank - 1;
	right = rank + 1;
	front = rank - xProcessNum;
	back = rank + xProcessNum;
	up = rank - xProcessNum * yProcessNum;
	down = rank + xProcessNum * yProcessNum;
	xDivisionSize = xBlockSize;
	yDivisionSize = yBlockSize;
	zDivisionSize = zBlockSize;

	if (rank % xProcessNum == 0)
		left = MPI_PROC_NULL;
	if ((rank + 1) % xProcessNum == 0)
	{
		right = MPI_PROC_NULL;
		xDivisionSize = (p.n1 - 2 * HALF_LENGTH) - (xProcessNum - 1) * xBlockSize;
	}
	if (rank % (xProcessNum * yProcessNum) < xProcessNum)
	{
		front = MPI_PROC_NULL;
	}
	if ((rank + xProcessNum) % (xProcessNum * yProcessNum) < xProcessNum)
	{
		back = MPI_PROC_NULL;
		yDivisionSize = (p.n2 - 2 * HALF_LENGTH) - (yProcessNum - 1) * yBlockSize;
	}
	if (up < 0)
	{
		up = MPI_PROC_NULL;
	}
	if (down >= pSize)
	{
		down = MPI_PROC_NULL;
		zDivisionSize = (p.n3 - 2 * HALF_LENGTH) - (yProcessNum - 1) * zBlockSize;
	}
	// allocate dat memory
	//printf("rank:%d left:%d right:%d up:%d down:%d \n", rank, left, right, up, down);
	size_t nsize = p.n1 * p.n2 * p.n3;

	size_t nsize_mpi = (2 * HALF_LENGTH + xDivisionSize) * (2 * HALF_LENGTH + yDivisionSize) * (2 * HALF_LENGTH + zDivisionSize);
	size_t nsize_xDimension_halo = HALF_LENGTH * (2 * HALF_LENGTH + yDivisionSize) * (2 * HALF_LENGTH + zDivisionSize); //左右两个halo区并成一块
	size_t nsize_yDimension_halo = (2 * HALF_LENGTH + xDivisionSize) * HALF_LENGTH * (2 * HALF_LENGTH + zDivisionSize);
	size_t nsize_zDimension_halo = (2 * HALF_LENGTH + xDivisionSize) * (2 * HALF_LENGTH + yDivisionSize) * HALF_LENGTH;
	size_t nbytes = nsize_mpi * sizeof(float);
	size_t nbytes_xDimension_halo = nsize_xDimension_halo * sizeof(float);
	size_t nbytes_yDimension_halo = nsize_yDimension_halo * sizeof(float);
	size_t nbytes_zDimension_halo = nsize_zDimension_halo * sizeof(float);

	//printf("nsize_mpi:%d ,xDivisionSize:%d ,yDivisionSize:%d \n", nsize_mpi, xDivisionSize, yDivisionSize);
	//printf("allocating prev, next and vel: total %g Mbytes\n", (3.0 * (nbytes + 16)) / (1024 * 1024));
	fflush(NULL);

	float *prev_base = (float *)_mm_malloc((nsize_mpi + 16 + MASK_ALLOC_OFFSET(0)) * sizeof(float), CACHELINE_BYTES);
	float *next_base = (float *)_mm_malloc((nsize_mpi + 16 + MASK_ALLOC_OFFSET(16)) * sizeof(float), CACHELINE_BYTES);
	float *vel_base = (float *)_mm_malloc((nsize_mpi + 16 + MASK_ALLOC_OFFSET(32)) * sizeof(float), CACHELINE_BYTES);

	float *ls = (float *)_mm_malloc((nbytes_xDimension_halo + 16 + MASK_ALLOC_OFFSET(0)) * sizeof(float), CACHELINE_BYTES);
	float *rs = (float *)_mm_malloc((nbytes_xDimension_halo + 16 + MASK_ALLOC_OFFSET(16)) * sizeof(float), CACHELINE_BYTES);
	float *fs = (float *)_mm_malloc((nbytes_yDimension_halo + 16 + MASK_ALLOC_OFFSET(32)) * sizeof(float), CACHELINE_BYTES);
	float *bs = (float *)_mm_malloc((nbytes_yDimension_halo + 16 + MASK_ALLOC_OFFSET(48)) * sizeof(float), CACHELINE_BYTES);
	float *us = (float *)_mm_malloc((nbytes_zDimension_halo + 16 + MASK_ALLOC_OFFSET(64)) * sizeof(float), CACHELINE_BYTES);
	float *ds = (float *)_mm_malloc((nbytes_zDimension_halo + 16 + MASK_ALLOC_OFFSET(80)) * sizeof(float), CACHELINE_BYTES);

	float *lr = (float *)_mm_malloc((nbytes_xDimension_halo + 16 + MASK_ALLOC_OFFSET(0)) * sizeof(float), CACHELINE_BYTES);
	float *rr = (float *)_mm_malloc((nbytes_xDimension_halo + 16 + MASK_ALLOC_OFFSET(16)) * sizeof(float), CACHELINE_BYTES);
	float *fr = (float *)_mm_malloc((nbytes_yDimension_halo + 16 + MASK_ALLOC_OFFSET(32)) * sizeof(float), CACHELINE_BYTES);
	float *br = (float *)_mm_malloc((nbytes_yDimension_halo + 16 + MASK_ALLOC_OFFSET(48)) * sizeof(float), CACHELINE_BYTES);
	float *ur = (float *)_mm_malloc((nbytes_zDimension_halo + 16 + MASK_ALLOC_OFFSET(64)) * sizeof(float), CACHELINE_BYTES);
	float *dr = (float *)_mm_malloc((nbytes_zDimension_halo + 16 + MASK_ALLOC_OFFSET(80)) * sizeof(float), CACHELINE_BYTES);

	if (prev_base == NULL || next_base == NULL || vel_base == NULL)
	{
		printf("couldn't allocate CPU memory prev_base=%p next=_base%p vel_base=%p\n", prev_base, next_base, vel_base);
		printf("  TEST FAILED!\n");
		fflush(NULL);
		exit(-1);
	}

	// Align working vectors offsets
	p.prev = &prev_base[16 + ALIGN_HALO_FACTOR + MASK_ALLOC_OFFSET(0)];
	p.next = &next_base[16 + ALIGN_HALO_FACTOR + MASK_ALLOC_OFFSET(16)];
	p.vel = &vel_base[16 + ALIGN_HALO_FACTOR + MASK_ALLOC_OFFSET(32)];

	p.leftSendBlock = &ls[16 + ALIGN_HALO_FACTOR + MASK_ALLOC_OFFSET(0)];
	p.rightSendBlock = &rs[16 + ALIGN_HALO_FACTOR + MASK_ALLOC_OFFSET(16)];
	p.frontSendBlock = &fs[16 + ALIGN_HALO_FACTOR + MASK_ALLOC_OFFSET(32)];
	p.backSendBlock = &bs[16 + ALIGN_HALO_FACTOR + MASK_ALLOC_OFFSET(48)];
	p.upSendBlock = &us[16 + ALIGN_HALO_FACTOR + MASK_ALLOC_OFFSET(64)];
	p.downSendBlock = &ds[16 + ALIGN_HALO_FACTOR + MASK_ALLOC_OFFSET(80)];

	p.leftRecvBlock = &lr[16 + ALIGN_HALO_FACTOR + MASK_ALLOC_OFFSET(0)];
	p.rightRecvBlock = &rr[16 + ALIGN_HALO_FACTOR + MASK_ALLOC_OFFSET(16)];
	p.frontRecvBlock = &fr[16 + ALIGN_HALO_FACTOR + MASK_ALLOC_OFFSET(32)];
	p.backRecvBlock = &br[16 + ALIGN_HALO_FACTOR + MASK_ALLOC_OFFSET(48)];
	p.upRecvBlock = &ur[16 + ALIGN_HALO_FACTOR + MASK_ALLOC_OFFSET(64)];
	p.downRecvBlock = &dr[16 + ALIGN_HALO_FACTOR + MASK_ALLOC_OFFSET(80)];

	//initialize(p.prev, p.next, p.vel, &p, nbytes);
	// A couple of run to start threading library
	//int tmp_nreps = 2;

	//iso_3dfd(p.next, p.prev, p.vel, coeff, p.n1, p.n2, p.n3, p.num_threads, tmp_nreps, p.n1_Tblock, p.n2_Tblock, p.n3_Tblock);

	int xOffSet = (rank % xProcessNum) * xBlockSize;
	int yOffSet = (rank % (xProcessNum * yProcessNum)) / xProcessNum * yBlockSize;
	int zOffSet = rank / (xProcessNum * yProcessNum) * zBlockSize;
	initiate_v6(rank, p.prev, p.next, p.vel, &p, xDivisionSize, yDivisionSize, zDivisionSize, xOffSet, yOffSet, zOffSet);
	wstart = walltime();
	//MPI_Type_vector(HALF_LENGTH+yDivisionSize+HALF_LENGTH, HALF_LENGTH, HALF_LENGTH + xDivisionSize + HALF_LENGTH, MPI_FLOAT, &yHaloType);
	//MPI_Type_commit(&yHaloType);
	//printf("initiate success\n");
	printf("rank:%d xs:%d ys:%d zs:%d\n",rank,xOffSet,yOffSet,zOffSet);
	int xSendRecvSize = HALF_LENGTH * (2 * HALF_LENGTH + yDivisionSize) * (2 * HALF_LENGTH + zDivisionSize);
	int ySendRecvSize = (2 * HALF_LENGTH + xDivisionSize) * HALF_LENGTH * (2 * HALF_LENGTH + zDivisionSize);
	int zSendRecvSize = (2 * HALF_LENGTH + xDivisionSize) * (2 * HALF_LENGTH + yDivisionSize) * HALF_LENGTH;
	for (int step = 0; step < /*p.nreps*/ 4; step++)
	{
		//void reference_implementation_v5(float *next, float *prev, float *coeff, float *vel, const int xDivisionSize, const int yDivisionSize, const int n3, const int half_length, const int xOffSet, const int yOffSet, const int rank)

		update_halo(&p, xDivisionSize, yDivisionSize, zDivisionSize, up, down, front, back, left, right);

		reference_implementation_v5(p.next, p.prev, coeff, p.vel, xDivisionSize, yDivisionSize, p.n3, HALF_LENGTH, xOffSet, yOffSet, rank);

		copy_data_to_send_block(&p, xDivisionSize, yDivisionSize, zDivisionSize);

		MPI_Sendrecv(&p.leftSendBlock[0], xSendRecvSize, MPI_FLOAT, left, 1, &p.rightRecvBlock[0], xSendRecvSize, MPI_FLOAT, right, 1, MPI_COMM_WORLD, &status);

		MPI_Sendrecv(&p.rightSendBlock[0], xSendRecvSize, MPI_FLOAT, right, 1, &p.leftRecvBlock[0], xSendRecvSize, MPI_FLOAT, left, 1, MPI_COMM_WORLD, &status);

		MPI_Sendrecv(&p.frontSendBlock[0], ySendRecvSize, MPI_FLOAT, front, 1, &p.backRecvBlock[0], ySendRecvSize, MPI_FLOAT, back, 1, MPI_COMM_WORLD, &status);

		MPI_Sendrecv(&p.backSendBlock[0], ySendRecvSize, MPI_FLOAT, back, 1, &p.frontRecvBlock[0], ySendRecvSize, MPI_FLOAT, front, 1, MPI_COMM_WORLD, &status);
	
		MPI_Sendrecv(&p.upSendBlock[0], zSendRecvSize, MPI_FLOAT, up, 1, &p.downRecvBlock[0], zSendRecvSize, MPI_FLOAT, down, 1, MPI_COMM_WORLD, &status);

		MPI_Sendrecv(&p.downSendBlock[0], zSendRecvSize, MPI_FLOAT, down, 1, &p.upRecvBlock[0], zSendRecvSize, MPI_FLOAT, up, 1, MPI_COMM_WORLD, &status);

		//update_halo(&p, xDivisionSize, yDivisionSize,up,down,left,right);

		float *temp;
		temp = p.next;
		p.next = p.prev;
		p.prev = temp;
	}
	
	// allocate dat memory
	//void output_v5(Parameters *p, int rank, int xDivisionSize, int yDivisionSize)
	//output_v6(&p, rank, xDivisionSize, yDivisionSize, zDivisionSize, xOffSet, yOffSet, zOffSet);

	//wstop = walltime();

	// report time
	//elapsed_time = wstop - wstart;
	//float normalized_time = elapsed_time / p.nreps;
	//throughput_mpoints = ((p.n1 - 2 * HALF_LENGTH) * (p.n2 - 2 * HALF_LENGTH) * (p.n3 - 2 * HALF_LENGTH)) / (normalized_time * 1e6f);
	//mflops = (7.0f * HALF_LENGTH + 5.0f) * throughput_mpoints;

	//printf("-------------------------------\n");
	//printf("time:       %8.2f sec\n", elapsed_time);
	//printf("throughput: %8.2f MPoints/s\n", throughput_mpoints);
	//printf("flops:      %8.2f GFlops\n", mflops / 1e3f);
#if defined(VERIFY_RESULTS)
	printf("\n-------------------------------\n");
	printf("comparing one iteration to reference implementation result...\n");

	initialize(p.prev, p.next, p.vel, &p, nbytes);

	p.nreps = 2;
	iso_3dfd(p.next, p.prev, p.vel, coeff, p.n1, p.n2, p.n3, p.num_threads, p.nreps, p.n1_Tblock, p.n2_Tblock, p.n3_Tblock);

	float *p_ref = (float *)malloc(p.n1 * p.n2 * p.n3 * sizeof(float));
	if (p_ref == NULL)
	{
		printf("couldn't allocate memory for p_ref\n");
		printf("  TEST FAILED!\n");
		fflush(NULL);
		exit(-1);
	}

	initialize(p.prev, p_ref, p.vel, &p, nbytes);
	reference_implementation(p_ref, p.prev, coeff, p.vel, p.n1, p.n2, p.n3, HALF_LENGTH);
	if (within_epsilon(p.next, p_ref, p.n1, p.n2, p.n3, HALF_LENGTH, 0, 0.0001f))
	{
		printf("  Result within epsilon\n");
		printf("  TEST PASSED!\n");
	}
	else
	{
		printf("  Incorrect result\n");
		printf("  TEST FAILED!\n");
	}
	free(p_ref);

#endif /* VERIFY_RESULTS */

	MPI_Finalize();
	_mm_free(prev_base);
	_mm_free(next_base);
	_mm_free(vel_base);
}
