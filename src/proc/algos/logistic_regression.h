#ifndef LOGISTIC_REGRESSION_H
#define LOGISTIC_REGRESSION_H

#include <vector>

using Matrix = std::vector<std::vector<double>>;
using Vector = std::vector<double>;

struct LogisticModel {
    Vector weights;
    double bias;
};

struct LogisticConfig {
    double learning_rate = 0.1;
    int epochs = 300;
    double l2 = 1e-3;
    int seed = 0;
    bool verbose = false;
};

LogisticModel logistic_regression_fit(const Matrix& X, const Vector& y,
                                     const LogisticConfig& config = LogisticConfig());

Vector logistic_regression_predict(const Matrix& X, const LogisticModel& model,
                                   double threshold = 0.5);

Vector logistic_regression_predict_proba(const Matrix& X, const LogisticModel& model);

#endif