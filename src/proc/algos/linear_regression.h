#ifndef LINEAR_REGRESSION_H
#define LINEAR_REGRESSION_H

#include <vector>

using Matrix = std::vector<std::vector<double>>;
using Vector = std::vector<double>;

struct LinearModel {
    Vector weights;
    double bias;
};

// Fit linear regression using closed-form solution (Normal Equation)
LinearModel linear_regression_fit(const Matrix& X, const Vector& y, double l2 = 0.0);

// Predict using linear regression model
Vector linear_regression_predict(const Matrix& X, const LinearModel& model);

#endif // LINEAR_REGRESSION_H