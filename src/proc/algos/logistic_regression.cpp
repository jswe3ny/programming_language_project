#include "logistic_regression.h"
#include <cmath>
#include <random>
#include <algorithm>
#include <iostream>

double sigmoid(double z) {
    z = std::max(-500.0, std::min(500.0, z));
    return 1.0 / (1.0 + std::exp(-z));
}

LogisticModel logistic_regression_fit(const Matrix& X, const Vector& y, 
                                     const LogisticConfig& config) {
    int n = X.size();
    int d = X[0].size();

    std::mt19937 rng(config.seed);
    std::normal_distribution<double> dist(0.0, 0.01);
    
    LogisticModel model;
    model.weights.resize(d);
    for (int j = 0; j < d; j++) {
        model.weights[j] = dist(rng);
    }
    model.bias = 0.0;

    for (int epoch = 0; epoch < config.epochs; epoch++) {
        Vector predictions(n);
        for (int i = 0; i < n; i++) {
            double z = model.bias;
            for (int j = 0; j < d; j++) {
                z += X[i][j] * model.weights[j];
            }
            predictions[i] = sigmoid(z);
        }

        Vector grad_w(d, 0.0);
        double grad_b = 0.0;
        
        for (int i = 0; i < n; i++) {
            double error = predictions[i] - y[i];
            grad_b += error;
            for (int j = 0; j < d; j++) {
                grad_w[j] += error * X[i][j];
            }
        }

        for (int j = 0; j < d; j++) {
            grad_w[j] = grad_w[j] / n + config.l2 * model.weights[j];
        }
        grad_b /= n;

        for (int j = 0; j < d; j++) {
            model.weights[j] -= config.learning_rate * grad_w[j];
        }
        model.bias -= config.learning_rate * grad_b;

        if (config.verbose && (epoch % 50 == 0 || epoch == config.epochs - 1)) {
            double loss = 0.0;
            for (int i = 0; i < n; i++) {
                loss += y[i] * std::log(predictions[i] + 1e-12) + 
                        (1 - y[i]) * std::log(1 - predictions[i] + 1e-12);
            }
            loss = -loss / n;
            std::cout << "Epoch " << epoch << ": Loss=" << loss << std::endl;
        }
    }
    
    return model;
}

Vector logistic_regression_predict_proba(const Matrix& X, const LogisticModel& model) {
    int n = X.size();
    Vector probabilities(n);
    
    for (int i = 0; i < n; i++) {
        double z = model.bias;
        for (size_t j = 0; j < model.weights.size() && j < X[i].size(); j++) {
            z += X[i][j] * model.weights[j];
        }
        probabilities[i] = sigmoid(z);
    }
    
    return probabilities;
}

Vector logistic_regression_predict(const Matrix& X, const LogisticModel& model, 
                                   double threshold) {
    Vector proba = logistic_regression_predict_proba(X, model);
    Vector predictions(proba.size());
    
    for (size_t i = 0; i < proba.size(); i++) {
        predictions[i] = (proba[i] >= threshold) ? 1.0 : 0.0;
    }
    
    return predictions;
}