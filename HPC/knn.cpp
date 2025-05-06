#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <map>
#include <omp.h>

struct Point {
    double x, y;
    int label;  // Class label
};

double distance(const Point& a, const Point& b) {
    return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2));
}

int classify(const std::vector<Point>& training_data, const Point& query, int k) {
    int n = training_data.size();
    std::vector<std::pair<double, int>> dist_label(n); // pair of (distance, label)

    // Parallel computation of distances
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        double d = distance(query, training_data[i]);
        dist_label[i] = { d, training_data[i].label };
    }

    // Sort based on distance
    std::sort(dist_label.begin(), dist_label.end());

    // Count the labels of k nearest neighbors
    std::map<int, int> label_count;
    for (int i = 0; i < k; ++i) {
        label_count[dist_label[i].second]++;
    }

    // Find the most frequent label
    int max_count = 0, result_label = -1;
    for (const auto& p : label_count) {
        if (p.second > max_count) {
            max_count = p.second;
            result_label = p.first;
        }
    }

    return result_label;
}
int main() {
    int num_train = 1000, num_test = 10, k = 3;
    std::vector<Point> training_data(num_train);
    std::vector<Point> test_data(num_test);

    // Random training data: two classes (0 and 1)
    srand(time(0));
    for (int i = 0; i < num_train; ++i) {
        training_data[i].x = rand() % 100;
        training_data[i].y = rand() % 100;
        training_data[i].label = rand() % 2;
    }

    // Random test points
    for (int i = 0; i < num_test; ++i) {
        test_data[i].x = rand() % 100;
        test_data[i].y = rand() % 100;
    }

    // Classify each test point
    double start = omp_get_wtime();
    for (int i = 0; i < num_test; ++i) {
        int predicted = classify(training_data, test_data[i], k);
        std::cout << "Test Point (" << test_data[i].x << ", " << test_data[i].y << ") -> Predicted Label: " << predicted << '\n';
    }
    double end = omp_get_wtime();
    std::cout << "\nTotal Classification Time: " << (end - start) << " seconds\n";

    return 0;
}