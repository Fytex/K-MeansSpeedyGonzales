#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>

#define N 10000000
#define K 4


typedef struct Point
{
    float x;
    float y;
} Point;

typedef struct ClustersInfo
{
    Point * acc_points;
    size_t * sizes;
} ClustersInfo;

typedef struct Output
{
    Point * clusters_center;
    size_t * clusters_size;
    size_t iterations;
} Output;



void init(Point * sample, Point * clusters_center)
{
    srand(10);

    for (size_t i = 0; i < N; i++)
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

    for (size_t i = 0; i < K; i++)
        clusters_center[i] = sample[i];
}

double get_sq_euclidean_dist(Point a, Point b)
{
    double dx = (a.x - b.x);
    double dy = (a.y - b.y);

    return dx * dx + dy * dy;
}

void cluster_points(Point * sample, Point * clusters_center, ClustersInfo clusters_info)
{
    float * dists = malloc(K * sizeof(float));

    for (size_t i = 0; i < N; i++)
    {
        Point point = sample[i];

        for (size_t j = 0; j < K; j++)
        {
            dists[j] = get_sq_euclidean_dist(clusters_center[j], point);
        }

        size_t best_cluster = 0;
        double best_cluster_dist = dists[0];

        for (size_t j = 1; j < K; j++)
        {
            if (dists[j] < best_cluster_dist)
            {
                best_cluster = j;
                best_cluster_dist = dists[j];
            }

        clusters_info.acc_points[best_cluster].x += point.x;
        clusters_info.acc_points[best_cluster].y += point.y;
        clusters_info.sizes[best_cluster]++;
        }
    }   

    free(dists);
}



Point * reevaluate_centers(ClustersInfo clusters_info)
{
    Point * new_clusters_center = malloc(K * sizeof(Point));

    for (size_t i = 0; i < K; i++)
        new_clusters_center[i] = (Point) {
            .x = clusters_info.acc_points[i].x / clusters_info.sizes[i],
            .y = clusters_info.acc_points[i].x / clusters_info.sizes[i]
        };

    return new_clusters_center;
}


int has_converged(Point * clusters_center, Point * new_clusters_center)
{
    int * used_list = calloc(K, sizeof(int));
    size_t success_cmp = 0;

    for (size_t i = 0; i < K; i++)
    {
        Point point = clusters_center[i];

        for (size_t j = 0; j < K; j++)
        {
            Point new_point = new_clusters_center[j];

            if (point.x == new_point.x && point.y == new_point.y && !used_list[j])
            {
                used_list[j] = 1;
                ++success_cmp;
                break;
            }
        }
        
    }

    free(used_list);
    return success_cmp == K;
}


Output find_centers(Point * sample, Point * clusters_center)
{
    int finished;
    size_t iterations = 0;

    Point * acc_points = malloc(K * sizeof(Point *));

    size_t * clusters_size = malloc(K * sizeof(size_t));

    ClustersInfo clusters_info =
    {
        .acc_points = acc_points,
        .sizes = clusters_size
    };

    do {
        memset(clusters_size, 0, K * sizeof(size_t));
        memset(acc_points, 0, K * sizeof(Point));

        cluster_points(sample, clusters_center, clusters_info);

        Point * new_clusters_center = reevaluate_centers(clusters_info);

        finished = has_converged(clusters_center, new_clusters_center);

        free(clusters_center);
        clusters_center = new_clusters_center;
        ++iterations;

    } while (!finished);

    iterations--; // Last iteration doesn't count because it's a verification

    free(clusters_info.acc_points);

    Output output = {
        .clusters_center = clusters_center,
        .clusters_size = clusters_size,
        .iterations = iterations
    };

    return output;
}


int main(void)
{
    Point * sample = malloc(N * sizeof(Point));
    Point * clusters_center = malloc(K * sizeof(Point));

    init(sample, clusters_center);

    Output output = find_centers(sample, clusters_center);

    printf("N = %d, K = %d\n", N, K);

    for (size_t i = 0; i < K; i++)
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