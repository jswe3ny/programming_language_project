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

// Fit logistic regression using gradient descent
LogisticModel logistic_regression_fit(const Matrix& X, const Vector& y,
                                     const LogisticConfig& config = LogisticConfig());

// Predict class labels (0 or 1)
Vector logistic_regression_predict(const Matrix& X, const LogisticModel& model,
                                   double threshold = 0.5);

// Predict probabilities
Vector logistic_regression_predict_proba(const Matrix& X, const LogisticModel& model);

#endif // LOGISTIC_REGRESSION_H