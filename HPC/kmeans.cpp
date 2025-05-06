#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <omp.h>

struct Point {
    double x, y;
    int cluster;
};

double distance(const Point& a, const Point& b) {
    return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2));
}

void kMeans(std::vector<Point>& points, int k, int max_iter = 100) {
    int n = points.size();
    std::vector<Point> centroids(k);

    // Initialize centroids randomly
    srand(time(0));
    for (int i = 0; i < k; ++i) centroids[i] = points[rand() % n];

    for (int iter = 0; iter < max_iter; ++iter) {
        // Assign clusters
        #pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            double min_dist = 1e9;
            for (int j = 0; j < k; ++j) {
                double d = distance(points[i], centroids[j]);
                if (d < min_dist) {
                    min_dist = d;
                    points[i].cluster = j;
                }
            }
        }

        // Update centroids
        std::vector<double> sum_x(k, 0), sum_y(k, 0);
        std::vector<int> count(k, 0);

        #pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            #pragma omp atomic
            sum_x[points[i].cluster] += points[i].x;
            #pragma omp atomic
            sum_y[points[i].cluster] += points[i].y;
            #pragma omp atomic
            count[points[i].cluster]++;
        }

        for (int j = 0; j < k; ++j) {
            if (count[j]) {
                centroids[j].x = sum_x[j] / count[j];
                centroids[j].y = sum_y[j] / count[j];
            }
        }
    }
}