#include "kmeans.h"

#define NUM_BLOCKS 78128
#define NUM_THREADS_PER_BLOCK 128
#define N NUM_BLOCKS*NUM_THREADS_PER_BLOCK // + x if wanted

#define K 32

using namespace std;



typedef struct Point
{
    float x;
    float y;
} Point;

typedef struct ClustersInfo
{
    float * sum_points_x;
    float * sum_points_y;
    int * sizes;
} ClustersInfo;

typedef struct Output
{
    Point * clusters_center;
    int * clusters_size;
    int iterations;
} Output;


void init(Point * sample, Point * clusters_center)
{
    srand(10);

    for (int i = 0; i < N; i++)
    {
        float x = (float) rand() / RAND_MAX;
        float y = (float) rand() / RAND_MAX;

        Point point = 
        {
            .x = x, 
            .y = y
        };

        sample[i] = point;
    }

    for (int i = 0; i < K; i++)
        clusters_center[i] = sample[i];
}

__device__
inline float get_sq_euclidean_dist(Point a, Point b)
{
    float dx = (a.x - b.x);
    float dy = (a.y - b.y);

    return dx * dx + dy * dy;
}


__global__
void kmeansKernel (Point * sample, Point * global_clusters_center, ClustersInfo clusters_info) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	int lid = threadIdx.x; // local thread id within a block


	__shared__ Point clusters_center[K];
    __shared__ float sum_points_x[K];
    __shared__ float sum_points_y[K];
    __shared__ int sizes[K];

    if (id < N)
    {

        if (lid == 0)
        {
            for (int j = 0; j < K; j++)
            {
                clusters_center[j] = global_clusters_center[j];
                sum_points_x[j] = 0.0f;
                sum_points_y[j] = 0.0f;
                sizes[j] = 0;
            }
        } 
        __syncthreads(); // wait for all threads within a block


        Point point = sample[id];

        int best_cluster = 0;

        float best_cluster_dist = get_sq_euclidean_dist(clusters_center[0], point);
        

        for (int j = 1; j < K; j++)
        {
            float cluster_dist = get_sq_euclidean_dist(clusters_center[j], point);

            if (cluster_dist < best_cluster_dist)
            {
                best_cluster = j;
                best_cluster_dist = cluster_dist;
            }
        }

        /*
        ARCH > 6.0
        atomicAdd_block(&sum_points_x[best_cluster], point.x);
        atomicAdd_block(&sum_points_y[best_cluster], point.y);
        atomicAdd_block(&sizes[best_cluster], 1);
        */


        atomicAdd(&sum_points_x[best_cluster], point.x);
        atomicAdd(&sum_points_y[best_cluster], point.y);
        atomicAdd(&sizes[best_cluster], 1);


        __syncthreads(); // wait for all threads within a block
        if (lid == 0)
        {
            for (int j = 0; j < K; j++)
            {
                atomicAdd(&clusters_info.sum_points_x[j], sum_points_x[j]);
                atomicAdd(&clusters_info.sum_points_y[j], sum_points_y[j]);
                atomicAdd(&clusters_info.sizes[j], sizes[j]);
            }
        }
    }
}


Point * reevaluate_centers(ClustersInfo clusters_info)
{
    Point * new_clusters_center = (Point *) malloc(K * sizeof(Point));

    for (int i = 0; i < K; i++)
        new_clusters_center[i] = (Point) {
            .x = clusters_info.sum_points_x[i] / clusters_info.sizes[i],
            .y = clusters_info.sum_points_y[i] / clusters_info.sizes[i],
        };


    return new_clusters_center;
}


int has_converged(Point * clusters_center, Point * new_clusters_center)
{
    return memcmp(clusters_center, new_clusters_center, K * sizeof(Point)) == 0 ? 1 : 0;
}


void initKmeansKernel(Point * sample, Point ** _device_sample, Point ** _device_clusters_center, ClustersInfo * _device_clusters_info)
{

    // declare variable with size of the array in bytes
	int sample_bytes = N * sizeof(Point);
    int clusters_bytes = K * sizeof(Point);
	int clusters_coord_bytes = K * sizeof(float);
	int clusters_sizes_bytes = K * sizeof(int);

    // pointers to the device memory
	Point * device_sample;
	Point * device_clusters_center;
	ClustersInfo device_clusters_info;

    // allocate the memory on the device
	cudaMalloc((void**) &device_sample, sample_bytes);
	cudaMalloc((void**) &device_clusters_center, clusters_bytes);
	cudaMalloc((void**) &device_clusters_info.sum_points_x, clusters_coord_bytes);
    cudaMalloc((void**) &device_clusters_info.sum_points_y, clusters_coord_bytes);
	cudaMalloc((void**) &device_clusters_info.sizes, clusters_sizes_bytes);
	checkCUDAError("mem allocation");

    // copy inputs to the device
	cudaMemcpy(device_sample, sample, sample_bytes, cudaMemcpyHostToDevice);
    checkCUDAError("memcpy h->d");

    // Return Values
    *_device_sample = device_sample;
	*_device_clusters_center = device_clusters_center;
	*_device_clusters_info = device_clusters_info;
}

void freeKmeansKernel(Point * device_sample, Point * device_clusters_center, ClustersInfo device_clusters_info)
{
    // free the device memory
	cudaFree(device_sample);
	cudaFree(device_clusters_center);
	cudaFree(device_clusters_info.sum_points_x);
    cudaFree(device_clusters_info.sum_points_y);
	cudaFree(device_clusters_info.sizes);
	checkCUDAError("mem free");
}


void launchKmeansKernel(Point * clusters_center, ClustersInfo clusters_info,
                        Point * device_sample, Point * device_clusters_center, ClustersInfo device_clusters_info)
{
	// declare variable with size of the array in bytes
    int clusters_bytes = K * sizeof(Point);
	int clusters_coord_bytes = K * sizeof(float);
	int clusters_sizes_bytes = K * sizeof(int);

    // reset inputs to the device
    cudaMemset(device_clusters_info.sum_points_x, 0.0f, clusters_coord_bytes);
    cudaMemset(device_clusters_info.sum_points_y, 0.0f, clusters_coord_bytes);
    cudaMemset(device_clusters_info.sizes, 0, clusters_sizes_bytes);

	// copy inputs to the device
	cudaMemcpy(device_clusters_center, clusters_center, clusters_bytes, cudaMemcpyHostToDevice);
	checkCUDAError("memcpy h->d");

	// launch the kernel
	//startKernelTime();
	kmeansKernel <<< NUM_BLOCKS, NUM_THREADS_PER_BLOCK >>> (device_sample, device_clusters_center, device_clusters_info);
	//stopKernelTime();
	checkCUDAError("kernel invocation");

	// copy the output to the host
	cudaMemcpy(clusters_info.sum_points_x, device_clusters_info.sum_points_x, clusters_coord_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(clusters_info.sum_points_y, device_clusters_info.sum_points_y, clusters_coord_bytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(clusters_info.sizes, device_clusters_info.sizes, clusters_sizes_bytes, cudaMemcpyDeviceToHost);
	checkCUDAError("memcpy d->h");
}

Output find_centers(Point * sample, Point * clusters_center)
{
    int finished;
    int iterations = 0;

    float * sum_points_x = (float *) malloc(K * sizeof(float));
    float * sum_points_y = (float *) malloc(K * sizeof(float));
    int * clusters_size = (int *) malloc(K * sizeof(int));

    ClustersInfo clusters_info =
    {
        .sum_points_x = sum_points_x,
        .sum_points_y = sum_points_y,
        .sizes = clusters_size
    };


    // pointers to the device memory
	Point * device_sample;
	Point * device_clusters_center;
	ClustersInfo device_clusters_info;

    initKmeansKernel(sample, &device_sample, &device_clusters_center, &device_clusters_info);


    do {
		launchKmeansKernel(clusters_center, clusters_info, device_sample, device_clusters_center, device_clusters_info);

        Point * new_clusters_center = reevaluate_centers(clusters_info);

        finished = has_converged(clusters_center, new_clusters_center);

        free(clusters_center);
        clusters_center = new_clusters_center;
        ++iterations;

    } while(!finished && iterations < 21);

    iterations--; // Last iteration doesn't count because it's a verification

    free(clusters_info.sum_points_x);
    free(clusters_info.sum_points_y);
    freeKmeansKernel(device_sample, device_clusters_center, device_clusters_info);

    Output output = {
        .clusters_center = clusters_center,
        .clusters_size = clusters_size,
        .iterations = iterations
    };

    return output;
}

int main(void)
{
	Point * sample = (Point *) malloc(N * sizeof(Point));
    Point * clusters_center = (Point *) malloc(K * sizeof(Point));

    init(sample, clusters_center);

	Output output = find_centers(sample, clusters_center);

    printf("N = %d, K = %d\n", N, K);

    for (int i = 0; i < K; i++)
    {
        printf("Center: (%.3f, %.3f) : Size: %d\n", 
            output.clusters_center[i].x,
            output.clusters_center[i].y,
            output.clusters_size[i]
        );
    }

    printf("Iterations: %d\n", output.iterations);

    free(output.clusters_center);
    free(output.clusters_size);
    

    return 0;
}
