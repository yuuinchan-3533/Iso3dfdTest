#include<stdio.h>
#include<math.h>
#include<mpi.h>



int rank, pSize; //rank：当前进程ID，pSize：总的进程数


void initialize(float* ptr_prev, float* ptr_vel, float* ptr_next,int x_size,int y_size,int z_size){  
    for (int k=0;k<z_size;k++){
        for (int j=0;j<y_size;j++){
            for (int i=0;i<x_size;i++){
                ptr_prev[z_size*y_size*k+y_size*j+i]=sin(k*100+j*10+i);
                ptr_vel[z_size*y_size*k+y_size*j+i]=cos(k*100+j*10+i);
                ptr_next[z_size*y_size*k+y_size*j+i]=0.00;
            }
        }
    }
}

int main(int argc, char* argv[]){

    int nthread;
    int x_size,y_size,z_size;
    int totalSize;
    int blockSize;
    int haloSize; 
    int k;
    int up,down;
    int kbegin,kend;
    blockSize=floor(z_size/nthread);//z方向上的分量
    haloSize=HALF_LENGTH;
    int status;
    x_size=300;
    y_size=300;
    z_size=256;
    nthread=12;

    if( (argc > 1) && (argc < 4) ) {
    	printf(" usage: [n1 n2 n3] [# threads] [# iterations] [thread block n1] [thread block n2] [thread block n3]\n");
    	exit(1);
  	}
  	// [n1 n2 n3]
  	if( argc >= 4 ) {
    		x_size = atoi(argv[1]);
    		y_size = atoi(argv[2]);
    		z_size = atoi(argv[3]);
  	}
  	//  [# threads]
  	if( argc >= 5){
    		nthread = atoi(argv[4]);
    }
    totalSize=x_size*y_size*z_size;
    
    float *prev = (float*)_mm_malloc( (totalSize+16+MASK_ALLOC_OFFSET(0 ))*sizeof(float), CACHELINE_BYTES);
  	float *vel = (float*)_mm_malloc( (totalSize+16+MASK_ALLOC_OFFSET(16))*sizeof(float), CACHELINE_BYTES);
  	float *next  = (float*)_mm_malloc( (totalSize+16+MASK_ALLOC_OFFSET(32))*sizeof(float), CACHELINE_BYTES);
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

    MPI_Init(&argc,&argv);//MPI初始化语句
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);//获取当前进程的pID
    MPI_Comm_size(MPI_COMM_WORLD,&pSize);//获取进程总数

     //（数据，数据大小，根进程编号，通讯域）将root进程的数据广播到所有其它的进程
    MPI_Bcast(COEFF, HALF_LENGTH+1, MPI_FLOAT, 0, MPI_COMM_WORLD);//（数据，数据大小，根进程编号，通讯域）
    initialize(pre_p,vel_p,next_p,x_size,y_size,z_size);
    
    if(rank<nthread){
        calculateBegin=HALF_LENGTH/2+k*blockSize;
        if(rank==0){
            up=MPI_PROC_NULL;
            down=k+1;
        }//根进程
        else if(k==nthread-1){
            up=k-1;
            down=MPI_PROC_NULL;
        }
        else{
            up=k-1;
            down=k+1;
        }

        for(int k=calculateBegin;k<calculateBegin+blockSize;k++){
            for(int i=HALF_LENGTH/2;i<x_size-HALF_LENGTH/2;i++){
                for (int j=HALF_LENGTH/2;j<y_size-HALF_LENGTH/2;j++){
                    float res = prev[k*z_size*y_size + j*y_size + i]*coeff[0];
	                
                    for(int ir=1; ir<=HALF_LENGTH; ir++) {
	                    res += coeff[ir] * (prev[(k+ir)*z_size*y_size + j*y_size + i] + prev[(k-ir)*z_size*y_size + j*y_size + i]);	      // horizontal
	                    res += coeff[ir] * (prev[k*z_size*y_size + (j+ir)*y_size + i] + prev[k*z_size*y_size + (j-ir)*y_size + i]);   // vertical
	                    res += coeff[ir] * (prev[k*z_size*y_size + j*y_size + i + ir] + prev[k*z_size*y_size + j*y_size + i - ir]); // in front / behind
	                }
	                next[k*z_size*y_size + j*y_size + i] = 2.0f* prev[k*z_size*y_size + j*y_size + i] - next[k*z_size*y_size + j*y_size + i] + res * vel[k*z_size*y_size + j*y_size + i];


                }   
            }
        }

        for(int k=calculateBegin;k<calculateBegin+blockSize;k++){
            for(int i=HALF_LENGTH/2;i<x_size-HALF_LENGTH/2;i++){
                for (int j=HALF_LENGTH/2;j<y_size-HALF_LENGTH/2;j++){
	                prev[k*z_size*y_size + j*y_size + i] = vel[k*z_size*y_size + j*y_size + i];
                    vel[k*z_size*y_size + j*y_size + i] = next[k*z_size*y_size + j*y_size + i];
                }   
            }
        }
        int upkey=calculateBegin*z_size*y_size + HALF_LENGTH/2*y_size + HALF_LENGTH/2;
        int downkey=(calculateBegin - 4)*z_size*y_size + HALF_LENGTH/2*y_size + calculateBegin - 4;
        
        //更新pre进程的下halo区,更新now进程的上halo区
        status=MPI_SENDRECV(vel[upkey]],4*x_size*y_size,MPI_FLOAT,up,1,vel[upkey],4*x_size*y_size,MPI_FLOAT,down,1,MPI_COMM_WORLD);
      

        //更新now进程的下halo区,更新next进程的上halo区
        status=MPI_SENDRECV(vel[downkey]],4*x_size*y_size,MPI_FLOAT,right,1,vel[downkey],4*x_size*y_size,MPI_FLOAT,down,1,MPI_COMM_WORLD);//上halo区
    }
    MPI_Finalize();
    return 0; 
}