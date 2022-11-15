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


float get_sq_euclidean_dist(Point a, Point b)
{
    float dx = (a.x - b.x);
    float dy = (a.y - b.y);

    return dx * dx + dy * dy;
}

void cluster_points(Point * restrict sample, Point * restrict clusters_center, ClustersInfo clusters_info)
{
    int * sizes = clusters_info.sizes;
    float * sum_points_x = clusters_info.sum_points_x;
    float * sum_points_y = clusters_info.sum_points_y;


    #pragma omp parallel for num_threads(NUM_THREADS) reduction(+:sum_points_x[:K]) reduction(+:sum_points_y[:K]) reduction(+:sizes[:K])
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

        sum_points_x[best_cluster] += point.x;
        sum_points_y[best_cluster] += point.y;
        sizes[best_cluster]++;
    }
}


Point * reevaluate_centers(ClustersInfo clusters_info)
{
    Point * new_clusters_center = malloc(K * sizeof(Point));

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


Output find_centers(Point * sample, Point * clusters_center)
{
    int finished;
    int iterations = 0;

    float * sum_points_x = malloc(K * sizeof(float));
    float * sum_points_y = malloc(K * sizeof(float));
    
    int * clusters_size = malloc(K * sizeof(int));

    ClustersInfo clusters_info =
    {
        .sum_points_x = sum_points_x,
        .sum_points_y = sum_points_y,
        .sizes = clusters_size
    };


    do {
        for (int j = 0; j < K; j++)
        {
            clusters_size[j] = 0;
            sum_points_x[j] = 0;
            sum_points_y[j] = 0;
        }

        cluster_points(sample, clusters_center, clusters_info);

        Point * new_clusters_center = reevaluate_centers(clusters_info);

        finished = has_converged(clusters_center, new_clusters_center);

        free(clusters_center);
        clusters_center = new_clusters_center;
        ++iterations;

    } while(!finished && iterations < 21);

    iterations--; // Last iteration doesn't count because it's a verification

    free(clusters_info.sum_points_x);
    free(clusters_info.sum_points_y);

    Output output = {
        .clusters_center = clusters_center,
        .clusters_size = clusters_size,
        .iterations = iterations
    };

    return output;
}


int main(int argc, char ** argv)
{
    if (argc < 3 || argc > 4)
    {
        printf("Valid command: ./k_means [Points] [Clusters] [Treads : Optional]");
        return -1;
    }

    N = atoi(argv[1]);
    K = atoi(argv[2]);
    NUM_THREADS = (argc == 4) ? atoi(argv[3]) : 1;

    Point * sample = malloc(N * sizeof(Point));
    Point * clusters_center = malloc(K * sizeof(Point));

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
