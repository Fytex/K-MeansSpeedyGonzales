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
    Point ** clusters;
    size_t * sizes;
} ClustersInfo;

typedef struct Output
{
    Point * clusters_center;
    size_t * clusters_size;
    size_t iterations;
} Output;



void init(Point * sample, size_t size, Point * clusters_center, size_t clusters_count)
{
    srand(10);

    for (size_t i = 0; i < size; i++)
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

    for (size_t i = 0; i < clusters_count; i++)
        clusters_center[i] = sample[i];
}

double get_sq_euclidean_dist(Point a, Point b)
{
    double dx = (a.x - b.x);
    double dy = (a.y - b.y);

    return dx * dx + dy * dy;
}

void cluster_points(Point * sample, size_t size, Point * clusters_center, size_t clusters_count, ClustersInfo clusters_info)
{
    for (size_t i = 0; i < size; i++)
    {
        Point point = sample[i];

        size_t best_cluster = 0;
        double best_cluster_dist = get_sq_euclidean_dist(clusters_center[0], point);
        

        for (size_t j = 1; j < clusters_count; j++)
        {
            double cluster_dist = get_sq_euclidean_dist(clusters_center[j], point);
            if (cluster_dist < best_cluster_dist)
            {
                best_cluster = j;
                best_cluster_dist = cluster_dist;
            }
        }

        clusters_info.clusters[best_cluster][clusters_info.sizes[best_cluster]] = point;
        clusters_info.sizes[best_cluster]++;
        
    }   
}


Point get_mean(Point * points, size_t count)
{
    float x = 0, y = 0; 

    for (size_t i = 0; i < count; i++)
    {
        x += points[i].x;
        y += points[i].y;
    }

    return (Point){
        .x = x / count,
        .y = y / count
    };
    
}


Point * reevaluate_centers(size_t clusters_count, ClustersInfo clusters_info)
{
    Point * new_clusters_center = malloc(clusters_count * sizeof(Point));

    for (size_t i = 0; i < clusters_count; i++)
        new_clusters_center[i] = get_mean(clusters_info.clusters[i], clusters_info.sizes[i]);

    return new_clusters_center;
}


int has_converged(Point * clusters_center, Point * new_clusters_center, size_t clusters_count)
{
    int * used_list = calloc(clusters_count, sizeof(int));
    size_t success_cmp = 0;

    for (size_t i = 0; i < clusters_count; i++)
    {
        Point point = clusters_center[i];

        for (size_t j = 0; j < clusters_count; j++)
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
    return success_cmp == clusters_count;
}


Output find_centers(Point * sample, size_t size, Point * clusters_center, size_t clusters_count)
{
    int finished;
    size_t iterations = 0;

    Point ** clusters = malloc(clusters_count * sizeof(Point *));
    for (size_t i = 0; i < clusters_count; i++)
        clusters[i] = malloc(size * sizeof(Point));
    
    size_t * clusters_size = malloc(clusters_count * sizeof(size_t));

    ClustersInfo clusters_info =
    {
        .clusters = clusters,
        .sizes = clusters_size
    };

    do {
        memset(clusters_size, 0, clusters_count * sizeof(size_t));

        cluster_points(sample, size, clusters_center, clusters_count, clusters_info);

        Point * new_clusters_center = reevaluate_centers(clusters_count, clusters_info);

        finished = has_converged(clusters_center, new_clusters_center, clusters_count);

        free(clusters_center);
        clusters_center = new_clusters_center;
        ++iterations;

    } while (!finished);

    iterations--; // Last iteration doesn't count because it's a verification

    free(clusters_info.clusters);

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

    init(sample, N, clusters_center, K);

    Output output = find_centers(sample, N, clusters_center, K);

    printf("N = %ld, K = %ld\n", N, K);

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