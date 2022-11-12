#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>

static int N;
static int K;
static int NUM_THREADS;


typedef struct Point
{
    float x;
    float y;
} Point;

typedef struct ThreadClustersInfo
{
    Point ** clusters;
    int * sizes;
} ThreadClustersInfo;


typedef struct ClustersInfo
{
    Point * sum_points;
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

ThreadClustersInfo *  generate_array_clusters_info(void)
{
    ThreadClustersInfo * array_clusters_info = malloc(NUM_THREADS * sizeof(ThreadClustersInfo));

    #pragma omp parallel for num_threads(NUM_THREADS)
    for (int thread_id = 0; thread_id < NUM_THREADS; thread_id++)
    {
        Point ** clusters = malloc(K * sizeof(Point *));
        for (size_t i = 0; i < K; i++)
            clusters[i] = malloc(N * sizeof(Point));
        

        int * clusters_size = malloc(K * sizeof(int));

        array_clusters_info[thread_id] = (ThreadClustersInfo)
        {
            .clusters = clusters,
            .sizes = clusters_size
        };
    }

    return array_clusters_info;
}

void init_array_clusters_info(ThreadClustersInfo * array_clusters_info)
{
    #pragma omp parallel for num_threads(NUM_THREADS)
    for (int thread_id = 0; thread_id < NUM_THREADS; thread_id++)
        memset(array_clusters_info[thread_id].sizes, 0, K * sizeof(int));
}

void free_array_clusters_info(ThreadClustersInfo * array_clusters_info)
{
    #pragma omp parallel for num_threads(NUM_THREADS)
    for (int thread_id = 0; thread_id < NUM_THREADS; thread_id++)
    {
        Point ** clusters =array_clusters_info[thread_id].clusters;
        for (size_t i = 0; i < K; i++)
            free(clusters[i]);

        free(clusters);
        

        free(array_clusters_info[thread_id].sizes);
    }

    free(array_clusters_info);
}

float get_sq_euclidean_dist(Point a, Point b)
{
    float dx = (a.x - b.x);
    float dy = (a.y - b.y);

    return dx * dx + dy * dy;
}

void cluster_points(Point * restrict sample, Point * restrict clusters_center, ThreadClustersInfo * restrict array_clusters_info)
{
    #pragma omp parallel num_threads(NUM_THREADS)
    {
        int thread_id = omp_get_thread_num();

        #pragma omp for
        for (int i = 0; i < N; i++)
        {
            Point point = sample[i];

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

            int idx = array_clusters_info[thread_id].sizes[best_cluster]++;

            array_clusters_info[thread_id].clusters[best_cluster][idx]= point;   
        }
    }
}


void reduction_clusters(ThreadClustersInfo * restrict array_clusters_info, ClustersInfo clusters_info)
{
    for (int thread_id = 0; thread_id < NUM_THREADS; thread_id++)
    {
        for (int j = 0; j < K; j++)
        {
            int size = array_clusters_info[thread_id].sizes[j];

            for (int idx = 0; idx < size; idx++)
            {
                clusters_info.sum_points[j].x += array_clusters_info[thread_id].clusters[j][idx].x;
                clusters_info.sum_points[j].y += array_clusters_info[thread_id].clusters[j][idx].y;
            }

            clusters_info.sizes[j] += size;
        }
    }
}


Point * reevaluate_centers(ClustersInfo clusters_info)
{
    Point * new_clusters_center = malloc(K * sizeof(Point));

    for (int i = 0; i < K; i++)
        new_clusters_center[i] = (Point) {
            .x = clusters_info.sum_points[i].x / clusters_info.sizes[i],
            .y = clusters_info.sum_points[i].y / clusters_info.sizes[i],
        };
        //new_clusters_center[i] = get_mean(clusters_info.clusters[i], clusters_info.sizes[i]);


    return new_clusters_center;
}



Output find_centers(Point * sample, Point * clusters_center)
{
    int finished;
    int iterations = 0;

    Point * sum_points = malloc(K * sizeof(Point));
    
    int * clusters_size = malloc(K * sizeof(int));

    ClustersInfo clusters_info =
    {
        .sum_points = sum_points,
        .sizes = clusters_size
    };

    ThreadClustersInfo * array_clusters_info = generate_array_clusters_info();

    for (int i = 0; i < 21; i++)
    {
        memset(clusters_size, 0, K * sizeof(int));
        memset(sum_points, 0, K * sizeof(Point));

        init_array_clusters_info(array_clusters_info);

        cluster_points(sample, clusters_center, array_clusters_info);

        reduction_clusters(array_clusters_info, clusters_info);

        Point * new_clusters_center = reevaluate_centers(clusters_info);

        //finished = has_converged(clusters_center, new_clusters_center);

        free(clusters_center);
        clusters_center = new_clusters_center;
        ++iterations;

    }

    iterations--; // Last iteration doesn't count because it's a verification

    free(clusters_info.sum_points);
    free_array_clusters_info(array_clusters_info);

    Output output = {
        .clusters_center = clusters_center,
        .clusters_size = clusters_size,
        .iterations = iterations
    };

    return output;
}


int main(int argc, char ** argv)
{
    if (argc != 4)
    {
        printf("Valid command: ./k_means 10000000 4 2");
        return -1;
    }

    N = atoi(argv[1]);
    K = atoi(argv[2]);
    NUM_THREADS = atoi(argv[3]);

    Point * sample = malloc(N * sizeof(Point));
    Point * clusters_center = malloc(K * sizeof(Point));

    init(sample, clusters_center);

    Output output = find_centers(sample, clusters_center);

    printf("N = %d, K = %d\n", N, K);

    for (int i = 0; i < K; i++)
    {
        printf("Center: (%.3f, %.3f) : Size: %ld\n", 
            output.clusters_center[i].x,
            output.clusters_center[i].y,
            output.clusters_size[i]
        );
    }

    printf("Iterations: %ld\n", output.iterations);

    free(output.clusters_center);
    free(output.clusters_size);
    

    return 0;
}
