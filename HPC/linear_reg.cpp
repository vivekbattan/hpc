#include <iostream>
#include <vector>
#include <omp.h>
using namespace std;

void linearRegression(const std::vector<double>& x, const std::vector<double>& y, double& m, double& b, double lr = 0.01, int epochs = 1000) {
    int n = x.size();
    for (int epoch = 0; epoch < epochs; ++epoch) {
        double m_grad = 0.0, b_grad = 0.0;

        #pragma omp parallel for reduction(+:m_grad, b_grad)
        for (int i = 0; i < n; ++i) {
            double pred = m * x[i] + b;
            double error = pred - y[i];
            m_grad += error * x[i];
            b_grad += error;
        }

        m -= lr * m_grad / n;
        b -= lr * b_grad / n;
    }
}

int main() {
    std::vector<double> x = {1, 2, 3, 4, 5};
    std::vector<double> y = {2, 4, 6, 8, 10};
    double m = 0.0, b = 0.0;

    linearRegression(x, y, m, b);

    std::cout << "Linear Regression: y = " << m << "x + " << b << "\n";
    return 0;
}

// void lin_reg(vector<int> &x, vector<int> &y, double m, double b, double lr=0.01, int epoch = 1000) {
//     int n = x.size();

//     for(int ep = 0; ep<epoch; ep++) {
//         double m_grad=0.0, b_grad=0.0;

//         #pragma omp parallel for reduction(+:, m_grad, b_grad)
//         for(int i=0;i<n;i++) {
//             double pred = m*x[i]+b;
//             double error = pred - y[i];
//             m_grad += error*x[i];
//             b_grad += error;
//         }

//         m-=lr*m_grad/n;
//         b-=lr*b_grad/n;
//     }
// }