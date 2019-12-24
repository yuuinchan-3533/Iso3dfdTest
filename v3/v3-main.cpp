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
int rank, pSize; //rank：当前进程ID，pSize：总的进程数

typedef struct{
	size_t n1;   	// First dimension
	size_t n2;   	// Second dimension
	size_t n3;   	// Third dimension
	int num_threads;
	int nreps;     	// number of time-steps, over which performance is averaged
	size_t n1_Tblock;	// Thread blocking on 1st dimension
	size_t n2_Tblock;	// Thread blocking on 2nd dimension
	size_t n3_Tblock;	// Thread blocking on 3rd dimension
	float *prev;	
	float *next;
	float *vel;
}Parameters; 

//Function used for initialization

void initialize_mpi(float* ptr_prev,float* ptr_next, float* ptr_vel,Parameters* p,size_t nbytes,int blockSize,int rank){
	
    for(int z=0;z<HALF_LENGTH+blockSize+HALF_LENGTH;z++){
        for(int y=0;y<p->n2;y++){
            for(int x=0;x<p->n1;x++){
                int offset=rank*(HALF_LENGTH+blockSize);
                int key=(z+offset)*p->n1*p->n2+y*p->n1+x;
                ptr_prev[key]=sin((z+offset)*100+y*10+x);
                ptr_next[key]=cos((z+offset)*100+y*10+x);
				ptr_vel[key] = 2250000.0f*DT*DT;//Integration of the v² and dt² here
            }
        }

    }
}
void output(Parameters* p,int blockSize,int rank){
	printf("blockSize:%d,rank:%d,HALF_LENGTH:%d\n",blockSize,rank,HALF_LENGTH);
	for(int z=HALF_LENGTH;z<HALF_LENGTH+blockSize;z++){
        for(int y=0;y<p->n2;y++){
            for(int x=0;x<p->n1;x++){
                int offset=rank*(HALF_LENGTH+blockSize);
                int key=(z+offset)*p->n1*p->n2+y*p->n1+x;
		printf("rank:%d(%d %d %d):%f\n",rank,x,y,z+offset,p->prev[key]);              
            }
        }

    }
}
void initialize(float* ptr_prev, float* ptr_next, float* ptr_vel, Parameters* p, size_t nbytes){
        memset(ptr_prev, 0.0f, nbytes);
        memset(ptr_next, 0.0f, nbytes);
        memset(ptr_vel, 1500.0f, nbytes);
        for(int i=0; i<p->n3; i++){
                for(int j=0; j<p->n2; j++){
                        for(int k=0; k<p->n1; k++){
                                ptr_prev[i*p->n2*p->n1 + j*p->n1 + k] = 0.0f;
                                ptr_next[i*p->n2*p->n1 + j*p->n1 + k] = 0.0f;
                                ptr_vel[i*p->n2*p->n1 + j*p->n1 + k] = 2250000.0f*DT*DT;//Integration of the v² and dt² here
                        }
                }
        }
	//Then we add a source
        float val = 1.f;
        for(int s=5; s>=0; s--){
                for(int i=p->n3/2-s; i<p->n3/2+s;i++){
                        for(int j=p->n2/4-s; j<p->n2/4+s;j++){
                                for(int k=p->n1/4-s; k<p->n1/4+s;k++){
                                        ptr_prev[i*p->n1*p->n2 + j*p->n1 + k] = val;
                                }
                        }
                }
                val *= 10;
       }
}

int main(int argc, char** argv)
{
	// Defaults
    MPI_Init(&argc, &argv);                //MPI初始化语句
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  //获取当前进程的pID
    MPI_Comm_size(MPI_COMM_WORLD, &pSize); //获取进程总数
	MPI_Status status;
	Parameters p;
	p.n1 = 256;   // First dimension x
  	p.n2 = 300;   // Second dimension y
  	p.n3 = 300;   // Third dimension z
  	p.num_threads = 24;
  	p.nreps = 100;     // number of time-steps, over which performance is averaged
  	p.n1_Tblock;       // Thread blocking on 1st dimension
  	p.n2_Tblock;       // Thread blocking on 2nd dimension
  	p.n3_Tblock;       // Thread blocking on 3rd dimension
# define N2_TBLOCK 1   // Default thread blocking on 2nd dimension: 1
# define N3_TBLOCK 124 // Default thread blocking on 3rd dimension: 124
	int up, down; // 相邻进程编号
    

  
  	if( (argc > 1) && (argc < 4) ) {
    		printf(" usage: [n1 n2 n3] [# threads] [# iterations] [thread block n1] [thread block n2] [thread block n3]\n");
    		exit(1);
  	}
  	// [n1 n2 n3]
  	if( argc >= 4 ) {
    		p.n1 = atoi(argv[1]);
    		p.n2 = atoi(argv[2]);
    		p.n3 = atoi(argv[3]);
  	}
  	//  [# threads]
  	if( argc >= 5)
    		p.num_threads = atoi(argv[4]);
  	//  [# iterations]
  	if( argc >= 6)
    		p.nreps = atoi(argv[5]);
  	//  [thread block n1] [thread block n2] [thread block n3]
  	if( argc >= 7) {
    		p.n1_Tblock = atoi(argv[6]);
  	} else {
    		p.n1_Tblock = p.n1; // Default: no blocking on 1st dim
  	}
  	if( argc >= 8) {
    		p.n2_Tblock = atoi(argv[7]);
  	} else {
    		p.n2_Tblock =  N2_TBLOCK;
  	}
  	if( argc >= 9) {
    		p.n3_Tblock = atoi(argv[8]);
  	} else {
    		p.n3_Tblock = N3_TBLOCK;
  	}
  
  	// Make sure n1 and n1_Tblock are multiple of 16 (to support 64B alignment)
  	if ((p.n1%16)!=0) {
    		printf("Parameter n1=%d must be a multiple of 16\n",p.n1);
    		exit(1);
  	}
  	if ((p.n1_Tblock%16)!=0) {
    		printf("Parameter n1_Tblock=%d must be a multiple of 16\n",p.n1_Tblock);
    		exit(1);
  	} 
  	// Make sure nreps is rouded up to next even number (to support swap)
  	p.nreps = ((p.nreps+1)/2)*2;



  	printf("n1=%d n2=%d n3=%d nreps=%d num_threads=%d HALF_LENGTH=%d\n",p.n1,p.n2,p.n3,p.nreps,p.num_threads,HALF_LENGTH);
  	printf("n1_thrd_block=%d n2_thrd_block=%d n3_thrd_block=%d\n", p.n1_Tblock, p.n2_Tblock, p.n3_Tblock);

    int blockSize=ceil((p.n3-2*HALF_LENGTH)/(pSize+0.0));

#if (HALF_LENGTH == 4)
        float coeff[HALF_LENGTH+1] = {
                        -2.847222222,
                        +1.6,
                        -0.2,
                        +2.53968e-2,

                        -1.785714e-3};
#elif (HALF_LENGTH == 8)
        float coeff[HALF_LENGTH+1] = {
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
#  error "HALF_LENGTH not implemented"
#endif
	//Apply the DX DY and DZ to coefficients
	coeff[0] = (3.0f*coeff[0]) / (DXYZ*DXYZ);
	for(int i=1; i<= HALF_LENGTH; i++){
		coeff[i] = coeff[i] / (DXYZ*DXYZ);
	}



  	// Data Arrays
  	p.prev=NULL, p.next=NULL, p.vel=NULL;

  	// variables for measuring performance
  	double wstart, wstop;
  	float elapsed_time=0.0f, throughput_mpoints=0.0f, mflops=0.0f;
    
  	// allocate dat memory
  	size_t nsize = p.n1*p.n2*p.n3;
	size_t nsize_mpi = (HALF_LENGTH+blockSize+HALF_LENGTH)*p.n1*p.n2;
  	size_t nbytes = nsize*sizeof(float);

  	printf("allocating prev, next and vel: total %g Mbytes\n",(3.0*(nbytes+16))/(1024*1024));fflush(NULL);

  	float *prev_base = (float*)_mm_malloc( (nsize+16+MASK_ALLOC_OFFSET(0 ))*sizeof(float), CACHELINE_BYTES);
  	float *next_base = (float*)_mm_malloc( (nsize+16+MASK_ALLOC_OFFSET(16))*sizeof(float), CACHELINE_BYTES);
  	float *vel_base  = (float*)_mm_malloc( (nsize+16+MASK_ALLOC_OFFSET(32))*sizeof(float), CACHELINE_BYTES);

  	if( prev_base==NULL || next_base==NULL || vel_base==NULL ){
    		printf("couldn't allocate CPU memory prev_base=%p next=_base%p vel_base=%p\n",prev_base, next_base, vel_base);
    		printf("  TEST FAILED!\n"); fflush(NULL);
    		exit(-1);
  	}

  	// Align working vectors offsets 
  	p.prev = &prev_base[16 +ALIGN_HALO_FACTOR +MASK_ALLOC_OFFSET(0 )];
  	p.next = &next_base[16 +ALIGN_HALO_FACTOR +MASK_ALLOC_OFFSET(16)];
  	p.vel  = &vel_base [16 +ALIGN_HALO_FACTOR +MASK_ALLOC_OFFSET(32)];

	//initialize(p.prev, p.next, p.vel, &p, nbytes);
  	// A couple of run to start threading library
  	//int tmp_nreps = 2;

  	//iso_3dfd(p.next, p.prev, p.vel, coeff, p.n1, p.n2, p.n3, p.num_threads, tmp_nreps, p.n1_Tblock, p.n2_Tblock, p.n3_Tblock);
	
	up = rank - 1;
    down = rank + 1;
    if (up == -1)
        up = MPI_PROC_NULL;
    if (down == pSize)
        down = MPI_PROC_NULL;

	if(rank=pSize-1){
		blockSize=p.n3-(pSize-1)*blockSize;
	}
	initialize_mpi(p.prev,p.next,p.vel,&p,2*HALF_LENGTH+blockSize,blockSize,rank);
  	wstart = walltime();
	for(int step=0;step<p.nreps;step++){
  		reference_implementation(p.next, p.prev, coeff, p.vel, p.n1, p.n2, p.n3, HALF_LENGTH );
		int nowrecvup = 0;
		int nowrecvdown = (blockSize+HALF_LENGTH)*p.n1*p.n2;
		int nowsend2up = HALF_LENGTH*p.n1*p.n2;
        int nowsend2down = blockSize*p.n1*p.n2;
		MPI_Sendrecv(&p.next[nowsend2up], HALF_LENGTH * p.n1 * p.n2, MPI_FLOAT, up, 1, &p.next[nowrecvdown], HALF_LENGTH * p.n1 * p.n2, MPI_FLOAT, down, 1, MPI_COMM_WORLD, &status);

        //更新now进程的下halo区,更新next进程的上halo区
        MPI_Sendrecv(&p.next[nowsend2down], HALF_LENGTH * p.n1 * p.n2, MPI_FLOAT, down, 1, &p.next[nowrecvup], HALF_LENGTH * p.n1 * p.n2, MPI_FLOAT, up, 1, MPI_COMM_WORLD, &status); //上halo区

		float *temp;
		temp=p.next;
		p.next=p.prev;
		p.prev=temp;	
	}
	printf("pSize:%d,x:%d,y:%d,z:%d\n",pSize,p.n1,p.n2,p.n3);
	output(&p,blockSize,rank);
	MPI_Finalize();
  	wstop =  walltime();

  	// report time
  	elapsed_time = wstop - wstart;
  	float normalized_time = elapsed_time/p.nreps;   
  	throughput_mpoints = ((p.n1-2*HALF_LENGTH)*(p.n2-2*HALF_LENGTH)*(p.n3-2*HALF_LENGTH))/(normalized_time*1e6f);
  	mflops = (7.0f*HALF_LENGTH + 5.0f)* throughput_mpoints;

  	printf("-------------------------------\n");
  	printf("time:       %8.2f sec\n", elapsed_time );
  	printf("throughput: %8.2f MPoints/s\n", throughput_mpoints );
  	printf("flops:      %8.2f GFlops\n", mflops/1e3f );
#if defined(VERIFY_RESULTS)
        printf("\n-------------------------------\n");
        printf("comparing one iteration to reference implementation result...\n");

        initialize(p.prev, p.next, p.vel, &p, nbytes);

        p.nreps=2;
  	iso_3dfd(p.next, p.prev, p.vel, coeff, p.n1, p.n2, p.n3, p.num_threads, p.nreps, p.n1_Tblock, p.n2_Tblock, p.n3_Tblock);

        float *p_ref = (float*)malloc(p.n1*p.n2*p.n3*sizeof(float));
        if( p_ref==NULL ){
                printf("couldn't allocate memory for p_ref\n");
                printf("  TEST FAILED!\n"); fflush(NULL);
                exit(-1);
        }

        initialize(p.prev, p_ref, p.vel, &p, nbytes);

        reference_implementation( p_ref, p.prev, coeff, p.vel, p.n1, p.n2, p.n3, HALF_LENGTH );
        if( within_epsilon( p.next, p_ref, p.n1, p.n2, p.n3, HALF_LENGTH, 0, 0.0001f ) ) {
                printf("  Result within epsilon\n");
                printf("  TEST PASSED!\n");
        } else {
                printf("  Incorrect result\n");
                printf("  TEST FAILED!\n");
        }
        free(p_ref);

#endif /* VERIFY_RESULTS */

  	_mm_free(prev_base);
  	_mm_free(next_base);
  	_mm_free(vel_base);
}
