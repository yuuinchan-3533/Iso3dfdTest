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
int rank, pSize;			  //rank：当前进程ID，pSize：总的进程数
int xProcessNum, yProcessNum; //x轴上划分的进程数、y轴上划分的进程数
int xBlockSize, yBlockSize;   //x轴每个进程算的格点数,y轴每个进程算得格点数

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
	float *prevHalo; //prev的halo区，prevHalo[0~HALF_LENGTH][y][z]为左halo区，prevHalo[HALF_LENGTH~2*HALF_LENGTH][y][z]为右halo区
	float *nextHalo;
	float *sendBlock; //需要被发送的数据区域，send[0~HALF_LENGTH][y][z]为向left传递的数据区，send[HALF_LENGTH~2*HALF_LENGTH][y][z]为向right传递的数据区
} Parameters;

//Function used for initialization

void output(Parameters *p, int blockSize, int rank)
{
	for (int rk = 0; rk < pSize; rk++)
	{
		fflush(stdout);
		MPI_Barrier(MPI_COMM_WORLD);
		if (rk == rank)
		{
			int offset = rank * (HALF_LENGTH + blockSize);
			for (int z = HALF_LENGTH; z < HALF_LENGTH + blockSize; z++)
			{
				for (int y = 0; y < p->n2; y++)
				{
					for (int x = 0; x < p->n1; x++)
					{
						int key = z * p->n1 * p->n2 + y * p->n1 + x;
						//printf("rank:%d(%d %d %d):%f\n",rank,x,y,z+offset,p->prev[key]);
						//printf("(%d %d %d):%.3f\n", x, y, z + offset, p->prev[key]);
					}
				}
			}
		}
	}
}

void output_2D(Parameters *p,int rank,int xDivisionSize,int yDivisionSize){
	int xOffSet = (rank % xProcessNum) * xBlockSize;
	int yOffSet = (rank / xProcessNum) * yBlockSize;
	int n2n3=(2*HALF_LENGTH+yDivisionSize)*p->n3;
	for (int rk = 0; rk < pSize; rk++){
                 fflush(stdout);
                 MPI_Barrier(MPI_COMM_WORLD);
                 if (rk == rank)
                 {
                         for (int x = HALF_LENGTH; x < HALF_LENGTH + xBlockSize; x++)
                         {
                                 for (int y = HALF_LENGTH; y < HALF_LENGTH + yBlockSize ; y++)
                                 {
                                         for (int z = 0; z < p->n3; z++)
                                         {
                                                 int key = x * n2n3 + y * p->n3 + z;
                                                 printf("rank:%d(%d %d %d):%f\n",rank,x+xOffSet,y+yOffSet,z,p->prev[key]);
                                                 //printf("(%d %d %d):%.3f\n", x, y, z + offset, p->prev[key]);
                                         }
                                 }
                         }
                 }
         }


}
void output_halo(const int n3,const int yDivisionSize,Parameters *p){
	printf("output_halo\n");
	int n2n3=(2*HALF_LENGTH+yDivisionSize)*n3;
	for(int ix=0;ix<2*HALF_LENGTH;ix++){
		for(int iy=0;iy<2*HALF_LENGTH+yDivisionSize;iy++){
			for(int iz=0;iz<n3;iz++){
//				printf("halo_rank:%d (%d %d %d) %.2f\n",rank,ix,iy,iz,p->nextHalo[ix]);
			}

		}
		return;

	}
}
void initialize(float *ptr_prev, float *ptr_next, float *ptr_vel, Parameters *p, size_t nbytes)
{
	memset(ptr_prev, 0.0f, nbytes);
	memset(ptr_next, 0.0f, nbytes);
	memset(ptr_vel, 1500.0f, nbytes);
	for (int i = 0; i < p->n3; i++)
	{
		for (int j = 0; j < p->n2; j++)
		{
			for (int k = 0; k < p->n1; k++)
			{
				ptr_prev[i * p->n2 * p->n1 + j * p->n1 + k] = 0.0f;
				ptr_next[i * p->n2 * p->n1 + j * p->n1 + k] = 0.0f;
				ptr_vel[i * p->n2 * p->n1 + j * p->n1 + k] = 2250000.0f * DT * DT; //Integration of the v² and dt² here
			}
		}
	}
	//Then we add a source
	float val = 1.f;
	for (int s = 5; s >= 0; s--)
	{
		for (int i = p->n3 / 2 - s; i < p->n3 / 2 + s; i++)
		{
			for (int j = p->n2 / 4 - s; j < p->n2 / 4 + s; j++)
			{
				for (int k = p->n1 / 4 - s; k < p->n1 / 4 + s; k++)
				{
					ptr_prev[i * p->n1 * p->n2 + j * p->n1 + k] = val;
				}
			}
		}
		val *= 10;
	}
}
void initiate_mpi_x_y(float *ptr_prev, float *ptr_next, float *ptr_vel, Parameters *p, int xDivisionSize, int yDivisionSize, int rank)
{
//	printf("%s %d\n",__FILE__,__LINE__);
	//printf("rank:%d xProcessNum:%d xBlockSize:%d\n",rank,xProcessNum,xBlockSize);
	int xOffSet = (rank % xProcessNum) * xBlockSize;
	int yOffSet = (rank / xProcessNum) * yBlockSize;
	int haloKey = -1;
	int n3=p->n3;
	int n2n3=(HALF_LENGTH + yDivisionSize + HALF_LENGTH)*n3;
//	printf("%s %d\n",__FILE__,__LINE__);
	
	for (int x = 0; x < HALF_LENGTH + xDivisionSize + HALF_LENGTH; x++)
	{
		for (int y = 0; y < HALF_LENGTH + yDivisionSize + HALF_LENGTH; y++)
		{
			for (int z = 0; z < p->n3; z++)
			{

				int key = x * n2n3 + y * n3 + z;
				ptr_prev[key] = sin((x + xOffSet) * 1 + (y + yOffSet) * 10 + z*100);
				ptr_next[key] = cos((x + xOffSet) * 1 + (y + yOffSet) * 10 + z*100);
				ptr_vel[key] = 2250000.0f * DT * DT;
				if(x < HALF_LENGTH){
					p->prevHalo[key] = ptr_prev[key];
					p->nextHalo[key] = ptr_next[key];

				}
				else if(x>=HALF_LENGTH&&x<xDivisionSize+HALF_LENGTH){
					p->sendBlock[(x - HALF_LENGTH) * n2n3 + y * n3 + z] = ptr_next[x * n2n3 + y * n3 + z];
					
				}
				//printf("prev:%.3f\n",ptr_next[x * n2n3 + y * n3 + z]);
			
				else if (x >= xDivisionSize&&x < xDivisionSize + HALF_LENGTH)
				{
					p->sendBlock[(x - xDivisionSize + HALF_LENGTH) * n2n3 + y * n3 + z] = ptr_next[x * n2n3 + y * n3 + z];
				}
				else{
					p->prevHalo[(x - xDivisionSize) * n2n3 + y * n3 + z] = ptr_prev[key];
					p->nextHalo[(x - xDivisionSize) * n2n3 + y * n3 + z] = ptr_next[key];
				}
//				printf("rank:%d (%d %d %d) prev:%.3f\n",rank,(x+xOffSet),(y+yOffSet),z,ptr_next[x * n2n3 + y * n3 + z]);
			
			}
		}
	}
//	printf("%s %d\n",__FILE__,__LINE__);
	
}
void initiate_params(int n1, int n2)
{
	int x, y;
	float diff = n1 > n2 ? (n1 + 0.0) / (n2 + 0.0) : (n2 + 0.0) / (n1 + 0.0);
	float minn = 100000.0;
	float tempdiff = 0;
	x = floor(sqrt(pSize));
	for (int i = x; i > 0; i--)
	{
		if (pSize % x == 0)
		{
			tempdiff = (pSize / x) / (x + 0.0);
			if (abs(tempdiff - diff) < minn)
			{
				minn = abs(tempdiff - diff);
				xProcessNum = x;
				yProcessNum = pSize / x;
			}
		}
	}
	xBlockSize = ceil((n1 - 2 * HALF_LENGTH) / (xProcessNum + 0.0));
	yBlockSize = ceil((n2 - 2 * HALF_LENGTH) / (yProcessNum + 0.0));
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
	p.nreps = 100;			   // number of time-steps, over which performance is averaged
	p.n1_Tblock;			   // Thread blocking on 1st dimension
	p.n2_Tblock;			   // Thread blocking on 2nd dimension
	p.n3_Tblock;			   // Thread blocking on 3rd dimension
#define N2_TBLOCK 1			   // Default thread blocking on 2nd dimension: 1
#define N3_TBLOCK 124		   // Default thread blocking on 3rd dimension: 124
	int up, down, left, right; // 相邻进程编号
	int blockSize;
	int xDivisionSize, yDivisionSize; //该进程在x轴上计算的空间大小、该进程在y轴上计算的空间大小
	MPI_Datatype yHaloType;			  //纵列

	printf("%s %d\n",__FILE__,__LINE__);
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

	printf("%s %d\n",__FILE__,__LINE__);
	
	initiate_params(p.n1, p.n2);
	//printf("%d  %d\n", xProcessNum, yProcessNum);
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


	printf("%s %d\n",__FILE__,__LINE__);
	// Data Arrays
	p.prev = NULL, p.next = NULL, p.vel = NULL;

	// variables for measuring performance
	double wstart, wstop;
	float elapsed_time = 0.0f, throughput_mpoints = 0.0f, mflops = 0.0f;

	left = rank - 1;
	right = rank + 1;
	up = rank + xProcessNum;
	down = rank - xProcessNum;
	xDivisionSize = xBlockSize;
	yDivisionSize = yBlockSize;
	
	printf("xblocksize:%d yblocksize:%d\n",xBlockSize,yBlockSize);

	printf("%s %d\n",__FILE__,__LINE__);
	
	if (rank % xProcessNum == 0)
		left = MPI_PROC_NULL;
	if ((rank + 1) % xProcessNum == 0)
	{
		right = MPI_PROC_NULL;
		xDivisionSize = (p.n1 - 2 * HALF_LENGTH) - (xProcessNum - 1) * xBlockSize;
	}
	if (up >= pSize)
	{
		up = MPI_PROC_NULL;
		yDivisionSize = (p.n2 - 2 * HALF_LENGTH) - (yProcessNum - 1) * yBlockSize;
	}
	if (down < 0)
	{
		down = MPI_PROC_NULL;
	}
	// allocate dat memory
	//printf("rank:%d left:%d right:%d up:%d down:%d \n", rank, left, right, up, down);
	size_t nsize = p.n1 * p.n2 * p.n3;
	size_t nsize_mpi = (2 * HALF_LENGTH + xDivisionSize) * (2 * HALF_LENGTH + yDivisionSize) * p.n3;
	size_t nsize_halo = 2 * HALF_LENGTH * (HALF_LENGTH + yDivisionSize + HALF_LENGTH) * p.n3; //左右两个halo区并成一块
	size_t nbytes = nsize_mpi * sizeof(float);
	size_t nbytes_halo = nsize_halo * sizeof(float);

	//printf("nsize_mpi:%d ,xDivisionSize:%d ,yDivisionSize:%d \n", nsize_mpi, xDivisionSize, yDivisionSize);
	//printf("allocating prev, next and vel: total %g Mbytes\n", (3.0 * (nbytes + 16)) / (1024 * 1024));
	fflush(NULL);

	float *prev_base = (float *)_mm_malloc((nsize_mpi + 16 + MASK_ALLOC_OFFSET(0)) * sizeof(float), CACHELINE_BYTES);
	float *next_base = (float *)_mm_malloc((nsize_mpi + 16 + MASK_ALLOC_OFFSET(16)) * sizeof(float), CACHELINE_BYTES);
	float *vel_base = (float *)_mm_malloc((nsize_mpi + 16 + MASK_ALLOC_OFFSET(32)) * sizeof(float), CACHELINE_BYTES);

	float *prev_halo = (float *)_mm_malloc((nbytes_halo + 16 + MASK_ALLOC_OFFSET(0)) * sizeof(float), CACHELINE_BYTES);
	float *next_halo = (float *)_mm_malloc((nbytes_halo + 16 + MASK_ALLOC_OFFSET(16)) * sizeof(float), CACHELINE_BYTES);
	float *send_block = (float *)_mm_malloc((nbytes_halo + 16 + MASK_ALLOC_OFFSET(32)) * sizeof(float), CACHELINE_BYTES);

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

	p.prevHalo = &prev_halo[16 + ALIGN_HALO_FACTOR + MASK_ALLOC_OFFSET(0)];
	p.nextHalo = &next_halo[16 + ALIGN_HALO_FACTOR + MASK_ALLOC_OFFSET(16)];
	p.sendBlock = &send_block[16 + ALIGN_HALO_FACTOR + MASK_ALLOC_OFFSET(32)];

	//initialize(p.prev, p.next, p.vel, &p, nbytes);
	// A couple of run to start threading library
	//int tmp_nreps = 2;

	//iso_3dfd(p.next, p.prev, p.vel, coeff, p.n1, p.n2, p.n3, p.num_threads, tmp_nreps, p.n1_Tblock, p.n2_Tblock, p.n3_Tblock);

	
	initiate_mpi_x_y(p.prev, p.next, p.vel, &p, xDivisionSize, yDivisionSize, rank);
	wstart = walltime();
	//MPI_Type_vector(HALF_LENGTH+yDivisionSize+HALF_LENGTH, HALF_LENGTH, HALF_LENGTH + xDivisionSize + HALF_LENGTH, MPI_FLOAT, &yHaloType);
	//MPI_Type_commit(&yHaloType);
	//printf("initiate success\n");
	int xOffSet = (rank % xProcessNum) * xBlockSize;
	int yOffSet = (rank / xProcessNum) * yBlockSize;
	
	for (int step = 0; step < /*p.nreps*/ 4; step++)
	{
		//reference_implementation_mpi_2D(float *next, float *prev, float *coeff, float *vel, float *preHalo,const int n3, const int half_length, const int xDivisionSize, const int yDivisionSize)
		reference_implementation_mpi_2D(p.next, p.prev, coeff, p.vel, p.prevHalo, p.n3, HALF_LENGTH, xDivisionSize, yDivisionSize,xOffSet,yOffSet,rank);
	
		//copy_next_to_senfloat *next, float *send, const int half_length, const int xDivisionSize,const int yDivisionSize, const int n3)
		copy_next_to_send(p.next, p.sendBlock, HALF_LENGTH, xDivisionSize, yDivisionSize, p.n3);

	
		int nowSend2Up = (HALF_LENGTH)*p.n3;
		int nowRecvUp = 0;

		int nowSend2Down = yDivisionSize * p.n3;
		int nowRecvDown = (HALF_LENGTH + yDivisionSize) * p.n3;

		int nowRecvLeft = 0;
		int nowSend2Left = 0;

		int nowRecvRight = (HALF_LENGTH) * (2 * HALF_LENGTH + yDivisionSize) * p.n3;
		int nowSend2Right = (HALF_LENGTH) * (2 * HALF_LENGTH + yDivisionSize) * p.n3;
		int haloSendSize = HALF_LENGTH * (HALF_LENGTH + yDivisionSize + HALF_LENGTH) * p.n3;

		//printf("send down success");
		MPI_Sendrecv(&p.next[nowSend2Up], HALF_LENGTH * p.n2 * p.n3, MPI_FLOAT, up, 1, &p.next[nowRecvDown], HALF_LENGTH * p.n2 * p.n3, MPI_FLOAT, down, 1, MPI_COMM_WORLD, &status);
//		printf("send up success\n");
		//更新now进程的下halo区,更新next进程的上halo区
		MPI_Sendrecv(&p.next[nowSend2Down], HALF_LENGTH * p.n2 * p.n3, MPI_FLOAT, down, 1, &p.next[nowRecvUp], HALF_LENGTH * p.n2 * p.n3, MPI_FLOAT, up, 1, MPI_COMM_WORLD, &status); //上halo区
//		printf("send down success\n");
		
		
		MPI_Sendrecv(&p.sendBlock[nowSend2Left], haloSendSize, MPI_FLOAT, left, 1, &p.nextHalo[nowRecvRight], haloSendSize, MPI_FLOAT, right, 1, MPI_COMM_WORLD, &status);
//		printf("send left success\n");
		MPI_Sendrecv(&p.sendBlock[nowSend2Right], haloSendSize, MPI_FLOAT, right, 1, &p.nextHalo[nowRecvLeft], haloSendSize, MPI_FLOAT, left, 1, MPI_COMM_WORLD, &status);
//		printf("send right success\n");
		

		float *temp;
		temp = p.next;
		p.next = p.prev;
		p.prev = temp;
	}
	output_2D(&p, rank,xDivisionSize,yDivisionSize);
	MPI_Finalize();
	wstop = walltime();

	// report time
	elapsed_time = wstop - wstart;
	float normalized_time = elapsed_time / p.nreps;
	throughput_mpoints = ((p.n1 - 2 * HALF_LENGTH) * (p.n2 - 2 * HALF_LENGTH) * (p.n3 - 2 * HALF_LENGTH)) / (normalized_time * 1e6f);
	mflops = (7.0f * HALF_LENGTH + 5.0f) * throughput_mpoints;

	printf("-------------------------------\n");
	printf("time:       %8.2f sec\n", elapsed_time);
	printf("throughput: %8.2f MPoints/s\n", throughput_mpoints);
	printf("flops:      %8.2f GFlops\n", mflops / 1e3f);
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

	_mm_free(prev_base);
	_mm_free(next_base);
	_mm_free(vel_base);
}
