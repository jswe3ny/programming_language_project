#ifndef NAIVE_BAYES_H
#define NAIVE_BAYES_H

#include <vector>
#include <map>

using Matrix = std::vector<std::vector<double>>;
using Vector = std::vector<double>;

struct GaussianNBModel {
    std::vector<double> classes;
    std::map<double, double> priors;
    std::map<double, Vector> means;
    std::map<double, Vector> variances;
};

struct GaussianNBConfig {
    double var_smoothing = 1e-9;
    std::map<double, double> prior_override;
};

GaussianNBModel gaussian_nb_fit(const Matrix& X, const Vector& y,
                               const GaussianNBConfig& config = GaussianNBConfig());

Vector gaussian_nb_predict(const Matrix& X, const GaussianNBModel& model);

#endif