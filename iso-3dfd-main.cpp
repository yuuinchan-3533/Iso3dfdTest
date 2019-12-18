#include<stdio.h>
#include<mpi.h>

#define MAX_ITERATION 100
#define X_MAX 256
#define Y_MAX 300
#define Z_MAX 300
#define HALF_LENGTH 8
#define DT 0.002
#define DXYZ 50.000
int rank, pSize; //rank：当前进程ID，pSize：总的进程数
float VAL_P[X_MAX*Y_MAX*Z_MAX];
float PRE_P[X_MAX*Y_MAX*Z_MAX];
float NEXT_P[X_MAX*Y_MAX*Z_MAX];
float COEFF[HALF_LENGTH+1] = {
                        -3.0548446,
                        +1.7777778,
                        -3.1111111e-1,
                        +7.572087e-2,
                        -1.76767677e-2,
                        +3.480962e-3,
                        -5.180005e-4,
                        +5.074287e-5,
                        -2.42812e-6};


void initialize(){
    
    for (int i=0;i<X_MAX;i++){
        for (int j=0;j<Y_MAX;j++){
            for (int k=0;k<Z_MAX;k++){
                VAL_P[X_MAX*Y_MAX*i+Y_MAX*j+k]=0.00;
                PRE_P[X_MAX*Y_MAX*i+Y_MAX*j+k]=0.00;
                NEXT_P[X_MAX*Y_MAX*i+Y_MAX*j+k]=0.00;
            }
        }
    }


}

int main(int argc, char* argv[]){
    int i,j,k,iterationStep;

    MPI_Init(&argc,&argv);//MPI初始化语句
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);//获取当前进程的pID
    MPI_Comm_size(MPI_COMM_WORLD,&pSize);//获取进程总数



    //（数据，数据大小，根进程编号，通讯域）将root进程的数据广播到所有其它的进程
    MPI_Bcast(COEFF, X_MAX*Y_MAX*Z_MAX, MPI_FLOAT, 0, MPI_COMM_WORLD);//（数据，数据大小，根进程编号，通讯域）
 
    initialize();

    int z=rank;//取pID=key为需要处理的数据下标
    iterationStep=0;
    if(k>=4||k<=Z_MAX-4){

     while(iterationStep<MAX_ITERATION){
        for(i=4;i<X_MAX-4;i++){
            for(j=4;j<Y_MAX-4;j++){
                k=rank;
                int key=i*X_MAX*Y_MAX+j*Y_MAX+k;
                float keyVal=VAL_P[key];
                float temp=0;
                int tempkey=0;
                float res=0;
                float val=0;
                for(int point=1;point<=4;point++){
                    tempkey=(i+point)*X_MAX*Y_MAX+j*Y_MAX+k;
                    temp+=VAL_P[tempkey];
                    tempkey=(i-point)*X_MAX*Y_MAX+j*Y_MAX+k;
                    temp+=VAL_P[tempkey];
                    tempkey=i*X_MAX*Y_MAX+(j+point)*Y_MAX+k;
                    temp+=VAL_P[tempkey];
                    tempkey=i*X_MAX*Y_MAX+(j-point)*Y_MAX+k;
                    temp+=VAL_P[tempkey];
                    tempkey=i*X_MAX*Y_MAX+j*Y_MAX+k+point;
                    temp+=VAL_P[tempkey];
                    tempkey=i*X_MAX*Y_MAX+j*Y_MAX+k-point;
                    temp+=VAL_P[tempkey];
                    val=val+COEFF[point]*temp;
                }
                res=2*PRE_P[key]-VAL_P[key]+DT/(DXYZ*DXYZ)*(3*COEFF[0]*COEFF[0]*VAL_P[key]+val);
                NEXT_P[tempkey]=res;

            }
        }
        MPI_Allreduce(VAL_P,PRE_P,X_MAX*Y_MAX*Z_MAX,MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD); 
        MPI_Allreduce(NEXT_P,VAL_P,X_MAX*Y_MAX*Z_MAX,MPI_FLOAT,MPI_SUM,MPI_COMM_WORLD); 
     }

    }
    MPI_Finalize();
    return 0;
    
    //step.1 根进程读取输入 
    //step.2 初始化矩阵
    //step.3 进行mpi初始化操作
    //step.4 根据pSize以及x,y,z大小进行block（带halo区）的分割
    //step.5 开始迭代
    //step.6 对于每个block进行计算，并且更新halo区
    //step.7 若迭代次数超过或误差小于err，则该进程结束迭代


}