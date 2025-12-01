#ifndef KNN_H
#define KNN_H

#include <vector>

using Matrix = std::vector<std::vector<double>>;
using Vector = std::vector<double>;

enum class DistanceMetric {
    EUCLIDEAN,
    MANHATTAN
};

enum class TieBreak {
    SMALLEST_LABEL,
    RANDOM
};

struct KNNConfig {
    int k = 5;
    DistanceMetric distance = DistanceMetric::EUCLIDEAN;
    bool weighted = false;
    TieBreak tie_break = TieBreak::SMALLEST_LABEL;
    int seed = 0;
    double eps = 1e-12;
};

Vector knn_predict(const Matrix& X_train, const Vector& y_train,
                  const Matrix& X_test, const KNNConfig& config = KNNConfig());

#endif