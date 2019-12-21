#include <stdio.h>
#include <math.h>
#include <mpi.h>

#define MASK_ALLOC_OFFSET(x) (x)
#define CACHELINE_BYTES 64
#define ITERATION 2
int rank, pSize; //rank：当前进程ID，pSize：总的进程数

void initialize(float *ptr_prev, float *ptr_vel, float *ptr_next, int x_size, int y_size, int blockSize, int haloSize)
{
    int z_size = haloSize + blockSize + haloSize;
    for (int k = 0; k < haloSize + blockSize + haloSize; k++)
    {
        for (int j = 0; j < y_size; j++)
        {
            for (int i = 0; i < x_size; i++)
            {   int key=x_size * y_size * k + x_size * j + i;
                ptr_prev[x_size * y_size * k + x_size * j + i] = 0.5;
                ptr_vel[x_size * y_size * k + x_size * j + i] = 2250000.0f * DT * DT;
                ptr_next[x_size * y_size * k + x_size * j + i] = 0.8;
                //printf("%f %f %f\n",ptr_prev[key],ptr_vel[key],ptr_next[key]);
            }
        }
    }
}

void outputMatrix(float *prt_vel, int haloSize, int blockSize, int x_size, int y_size, int z_size)
{
    //freopen("matrix.out", "w", stdout);
    for (int k = haloSize; k < haloSize + blockSize; k++)
    {
        for (int j = HALF_LENGTH; j < y_size - HALF_LENGTH; j++)
        {
            for (int i = HALF_LENGTH; i < x_size - HALF_LENGTH; i++)
            {

                //prt_vel[k * x_size * y_size + j * x_size + i] = next[k * z_size * y_size + j * y_size + i];
                printf("%f ", prt_vel[k * x_size * y_size + j * x_size + i]);
            }
            printf("\n");
        }
        printf("\n");
    }
    //fclose(stdout);
}

int main(int argc, char *argv[])
{

    int nthread;
    int x_size, y_size, z_size;
    int totalSize;
    int blockSize;
    int haloSize;
    int k;
    int up, down;
    int kbegin, kend;
    int step;
//    printf("%s,%d\n", __FILE__, __LINE__);
    MPI_Init(&argc, &argv);                //MPI初始化语句
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  //获取当前进程的pID
    MPI_Comm_size(MPI_COMM_WORLD, &pSize); //获取进程总数
    haloSize = HALF_LENGTH;
    MPI_Status status;
    x_size = 100;
    y_size = 100;
    z_size = 64;
    nthread = pSize;

    if ((argc > 1) && (argc < 4))
    {
        //printf(" usage: [n1 n2 n3] [# threads] [# iterations] [thread block n1] [thread block n2] [thread block n3]\n");
        exit(1);
    }
    // [n1 n2 n3]
    if (argc >= 4)
    {
        x_size = atoi(argv[1]);
        y_size = atoi(argv[2]);
        z_size = atoi(argv[3]);
    }
    //  [# threads]
    if (argc >= 5)
    {
        nthread = atoi(argv[4]);
    }
    blockSize = floor(z_size / nthread); //z方向上的分量
    totalSize = x_size * y_size * (haloSize + blockSize + haloSize);
    //printf("x_size:%d,y_size:%d,z_size:%d,nthread:%d\n", x_size, y_size, z_size, nthread);
    float *prev = (float *)_mm_malloc((totalSize + 16 + MASK_ALLOC_OFFSET(0)) * sizeof(float), CACHELINE_BYTES);
    float *vel = (float *)_mm_malloc((totalSize + 16 + MASK_ALLOC_OFFSET(16)) * sizeof(float), CACHELINE_BYTES);
    float *next = (float *)_mm_malloc((totalSize + 16 + MASK_ALLOC_OFFSET(32)) * sizeof(float), CACHELINE_BYTES);
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

    //printf("1.进行mpi初始化");

    coeff[0] = (3.0f * coeff[0]) / (DXYZ * DXYZ);
    for (int i = 1; i <= HALF_LENGTH; i++)
    {
        coeff[i] = coeff[i] / (DXYZ * DXYZ);
    }
    //printf("%s,%d\n", __FILE__, __LINE__);
    //（数据，数据大小，根进程编号，通讯域）将root进程的数据广播到所有其它的进程
    //MPI_Bcast(coeff, HALF_LENGTH + 1, MPI_FLOAT, 0, MPI_COMM_WORLD); //（数据，数据大小，根进程编号，通讯域）
    initialize(prev, vel, next, x_size, y_size, blockSize, haloSize);
    step = 0;
    while (step < ITERATION)
    {
        int calculateBegin = HALF_LENGTH + k * blockSize;
        up = rank - 1;
        down = rank + 1;
        if (up == -1)
            up = MPI_PROC_NULL;
        if (down == nthread)
            down = MPI_PROC_NULL;

        //            if (rank == 0)
        //            {
        //                up = MPI_PROC_NULL;
        //                down = k + 1;
        //            } //根进程
        //            else if (k == nthread - 1)
        //            {
        //                up = k - 1;
        //                down = MPI_PROC_NULL;
        //            }
        //            else
        //            {
        //                up = k - 1;
        //                down = k + 1;
        //            }
        //printf("2.开始迭代计算");
        for (int k = haloSize; k < haloSize + blockSize; k++)
        {
            for (int j = HALF_LENGTH; j < y_size - HALF_LENGTH; j++)
            {
                for (int i = HALF_LENGTH; i < x_size - HALF_LENGTH; i++)
                {
                    float res = prev[k * x_size * y_size + j * x_size + i] * coeff[0];

                    for (int ir = 1; ir <= HALF_LENGTH; ir++)
                    {
                        res += coeff[ir] * (prev[(k + ir) * x_size * y_size + j * x_size + i] + prev[(k - ir) * x_size * y_size + j * x_size + i]); // horizontal
                        res += coeff[ir] * (prev[k * x_size * y_size + (j + ir) * x_size + i] + prev[k * x_size * y_size + (j - ir) * x_size + i]); // vertical
                        res += coeff[ir] * (prev[k * x_size * y_size + j * x_size + i + ir] + prev[k * x_size * y_size + j * x_size + i - ir]);     // in front / behind
                    }
                    next[k * x_size * y_size + j * x_size + i] = 2.0f * prev[k * x_size * y_size + j * x_size + i] - next[k * x_size * y_size + j * x_size + i] + res * vel[k * x_size * y_size + j * x_size + i];
                }
            }
        }
        //printf("3.开始赋值计算");
        //printf("%s,%d\n", __FILE__, __LINE__);
        for (int k = haloSize; k < haloSize + blockSize; k++)
        {
            for (int j = HALF_LENGTH; j < y_size - HALF_LENGTH; j++)
            {
                for (int i = HALF_LENGTH; i < x_size - HALF_LENGTH; i++)
                {
                    prev[k * x_size * y_size + j * x_size + i] = vel[k * x_size * y_size + j * x_size + i];
                    vel[k * x_size * y_size + j * x_size + i] = next[k * x_size * y_size + j * x_size + i];
                    //printf("%f\n", vel[k * x_size * y_size + j * x_size + i]);
                }
            }
        }

        //printf("%s,%d\n", __FILE__, __LINE__);

        int sendhalo1pos = haloSize * x_size * y_size + HALF_LENGTH * x_size + HALF_LENGTH;
        int recvhalo2pos = (haloSize + blockSize) * x_size * y_size + HALF_LENGTH * x_size + HALF_LENGTH;
        int sendhalo2pos = blockSize * x_size * y_size + HALF_LENGTH * x_size + HALF_LENGTH;
        int recvhalo1pos = HALF_LENGTH * x_size + HALF_LENGTH;

        //printf("开始更新halo区");
        //更新pre进程的下halo区,更新now进程的上halo区
        //printf("%s,%d\n", __FILE__, __LINE__);

        if (rank == 0)
            //printf("rank 0 send:%d,%d,%d\n", sendhalo1pos, haloSize * x_size * y_size, up);
        if (rank == 0)
            ///printf("rank 0 recv:%d,%d,%d\n", recvhalo2pos, haloSize * x_size * y_size, down);

        if (rank == 1)
            //printf("rank 1 send:%d,%d,%d\n", sendhalo1pos, haloSize * x_size * y_size, up);
        if (rank == 1)
            //printf("rank 1 recv:%d,%d,%d\n", recvhalo2pos, haloSize * x_size * y_size, down);

        MPI_Sendrecv(&vel[sendhalo1pos], haloSize * x_size * y_size, MPI_FLOAT, up, 1, &vel[recvhalo2pos], haloSize * x_size * y_size, MPI_FLOAT, down, 1, MPI_COMM_WORLD, &status);

        //更新now进程的下halo区,更新next进程的上halo区
        MPI_Sendrecv(&vel[sendhalo2pos], haloSize * x_size * y_size, MPI_FLOAT, down, 1, &vel[recvhalo1pos], haloSize * x_size * y_size, MPI_FLOAT, up, 1, MPI_COMM_WORLD, &status); //上halo区
        //printf("%s,%d\n", __FILE__, __LINE__);
        step++;
    }
    outputMatrix(vel, haloSize, blockSize, x_size, y_size, z_size);
    MPI_Finalize();
    return 0;
}
