#include "naive_bayes.h"
#include <cmath>
#include <set>
#include <algorithm>
#include <limits>

double gaussian_log_pdf(double x, double mean, double var, double eps = 1e-9) {
    var = std::max(var, eps);
    return -0.5 * std::log(2 * M_PI * var) - 0.5 * std::pow(x - mean, 2) / var;
}

GaussianNBModel gaussian_nb_fit(const Matrix& X, const Vector& y,
                               const GaussianNBConfig& config) {
    GaussianNBModel model;

    // Find unique classes
    std::set<double> classes_set(y.begin(), y.end());
    model.classes.assign(classes_set.begin(), classes_set.end());

    int n = X.size();
    int d = X[0].size();

    std::map<double, double> normalized_priors;
    if (!config.prior_override.empty()) {
        double sum = 0.0;
        for (double c : model.classes) {
            auto it = config.prior_override.find(c);
            if (it != config.prior_override.end()) {
                sum += it->second;
            }
        }

        if (sum > 0) {
            for (double c : model.classes) {
                auto it = config.prior_override.find(c);
                if (it != config.prior_override.end()) {
                    normalized_priors[c] = it->second / sum;
                }
            }
        }
    }

    // Compute statistics for each class
    for (double c : model.classes) {
        Matrix X_c;
        for (int i = 0; i < n; i++) {
            if (std::abs(y[i] - c) < 1e-9) {
                X_c.push_back(X[i]);
            }
        }

        int n_c = X_c.size();

        if (!normalized_priors.empty() && normalized_priors.count(c) > 0) {
            model.priors[c] = normalized_priors[c];
        } else {
            model.priors[c] = (double)n_c / n;
        }

        Vector means(d, 0.0);
        for (int j = 0; j < d; j++) {
            for (int i = 0; i < n_c; i++) {
                means[j] += X_c[i][j];
            }
            means[j] /= n_c;
        }
        model.means[c] = means;

        Vector vars(d, 0.0);
        for (int j = 0; j < d; j++) {
            for (int i = 0; i < n_c; i++) {
                double diff = X_c[i][j] - means[j];
                vars[j] += diff * diff;
            }
            vars[j] = vars[j] / n_c + config.var_smoothing;
        }
        model.variances[c] = vars;
    }

    return model;
}

Vector gaussian_nb_predict(const Matrix& X, const GaussianNBModel& model) {
    int n = X.size();
    int d = X[0].size();
    Vector predictions(n);

    for (int i = 0; i < n; i++) {
        double best_log_prob = -std::numeric_limits<double>::infinity();
        double best_class = model.classes[0];

        // Compute log probability for each class
        for (double c : model.classes) {
            double log_prob = std::log(model.priors.at(c) + 1e-12);

            const Vector& means = model.means.at(c);
            const Vector& vars = model.variances.at(c);

            // Sum log likelihoods for all features
            for (int j = 0; j < d; j++) {
                log_prob += gaussian_log_pdf(X[i][j], means[j], vars[j]);
            }

            if (log_prob > best_log_prob) {
                best_log_prob = log_prob;
                best_class = c;
            }
        }

        predictions[i] = best_class;
    }

    return predictions;
}