#ifndef METRICS_H
#define METRICS_H

#include <vector>

using Vector = std::vector<double>;

// Classification metrics
double accuracy(const Vector& y_true, const Vector& y_pred);
double macro_f1(const Vector& y_true, const Vector& y_pred);

// Regression metrics
double rmse(const Vector& y_true, const Vector& y_pred);
double r2_score(const Vector& y_true, const Vector& y_pred);

#endif // METRICS_H