#include "knn.h"
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include <random>
#include <limits>

inline double euclidean_distance(const Vector& a, const Vector& b) {
    double sum = 0.0;
    const size_t n = a.size();
    for (size_t i = 0; i < n; i++) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

inline double manhattan_distance(const Vector& a, const Vector& b) {
    double sum = 0.0;
    const size_t n = a.size();
    for (size_t i = 0; i < n; i++) {
        sum += std::abs(a[i] - b[i]);
    }
    return sum;
}

Vector knn_predict(const Matrix& X_train, const Vector& y_train,
                  const Matrix& X_test, const KNNConfig& config)
{
    const int n_test = X_test.size();
    const int n_train = X_train.size();
    Vector predictions(n_test);

    // Pre-allocate distance buffers
    std::vector<double> dists(n_train);
    std::vector<double> labels(n_train);

    std::mt19937 rng(config.seed);
    auto metric = (config.distance == DistanceMetric::EUCLIDEAN)
        ? euclidean_distance
        : manhattan_distance;

    for (int i = 0; i < n_test; i++) {
        const auto& x = X_test[i];

        // Compute all distances
        for (int j = 0; j < n_train; j++) {
            dists[j] = metric(x, X_train[j]);
            labels[j] = y_train[j];
        }

        // nth_element places the k smallest distances at the front (unordered)
        int k = std::min(config.k, n_train);
        std::nth_element(
            dists.begin(),
            dists.begin() + k,
            dists.end(),
            [&](double a, double b){ return a < b; }
        );

        // Sort only the first k items to align with labels
        std::vector<std::pair<double,double>> neighbors(k);
        for (int j = 0; j < k; j++)
            neighbors[j] = {dists[j], labels[j]};

        std::sort(neighbors.begin(), neighbors.end(),
                  [](auto &a, auto &b){ return a.first < b.first; });

        // Vote
        std::unordered_map<double, double> score;
        score.reserve(k);

        for (int j = 0; j < k; j++) {
            double dist = neighbors[j].first;
            double label = neighbors[j].second;

            double w = config.weighted
                ? 1.0 / (dist + config.eps)
                : 1.0;

            score[label] += w;
        }

        // Find best label
        double best_label = std::numeric_limits<double>::max();
        double best_score = -1.0;

        std::vector<double> tied;

        for (auto& kv : score) {
            double label = kv.first;
            double s = kv.second;

            if (s > best_score) {
                best_score = s;
                best_label = label;
                tied.clear();
                tied.push_back(label);
            }
            else if (std::abs(s - best_score) < 1e-12) {
                tied.push_back(label);
            }
        }

        // Tie break
        if (tied.size() > 1) {
            if (config.tie_break == TieBreak::SMALLEST_LABEL)
                best_label = *std::min_element(tied.begin(), tied.end());
            else {
                std::uniform_int_distribution<int> dis(0, tied.size()-1);
                best_label = tied[dis(rng)];
            }
        }

        predictions[i] = best_label;
    }

    return predictions;
}