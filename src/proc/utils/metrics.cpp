#include <cmath>
#include <set>
#include <algorithm>
#include <vector>

using Vector = std::vector<double>;

double accuracy(const Vector& y_true, const Vector& y_pred) {
    if (y_true.size() != y_pred.size() || y_true.empty()) return 0.0;
    
    int correct = 0;
    for (size_t i = 0; i < y_true.size(); i++) {
        if (std::abs(y_true[i] - y_pred[i]) < 1e-9) {
            correct++;
        }
    }
    return (double)correct / y_true.size();
}

double _f1_for_label(const Vector& y_true, const Vector& y_pred, double label) {
    double tp = 0, fp = 0, fn = 0;
    
    for (size_t i = 0; i < y_true.size(); i++) {
        bool true_pos = (std::abs(y_true[i] - label) < 1e-9);
        bool pred_pos = (std::abs(y_pred[i] - label) < 1e-9);
        
        if (true_pos && pred_pos) tp++;
        else if (!true_pos && pred_pos) fp++;
        else if (true_pos && !pred_pos) fn++;
    }
    
    double precision = tp / (tp + fp + 1e-12);
    double recall = tp / (tp + fn + 1e-12);
    return 2.0 * precision * recall / (precision + recall + 1e-12);
}

double macro_f1(const Vector& y_true, const Vector& y_pred) {
    std::set<double> labels_set(y_true.begin(), y_true.end());
    std::vector<double> labels(labels_set.begin(), labels_set.end());
    
    double sum_f1 = 0.0;
    for (double label : labels) {
        sum_f1 += _f1_for_label(y_true, y_pred, label);
    }
    
    return sum_f1 / labels.size();
}

double rmse(const Vector& y_true, const Vector& y_pred) {
    if (y_true.size() != y_pred.size() || y_true.empty()) return 0.0;
    
    double sum_sq = 0.0;
    for (size_t i = 0; i < y_true.size(); i++) {
        double diff = y_true[i] - y_pred[i];
        sum_sq += diff * diff;
    }
    
    return std::sqrt(sum_sq / y_true.size());
}

double r2_score(const Vector& y_true, const Vector& y_pred) {
    if (y_true.size() != y_pred.size() || y_true.empty()) return 0.0;
    
    // Compute mean
    double mean = 0.0;
    for (double y : y_true) mean += y;
    mean /= y_true.size();
    
    // Total sum of squares
    double sst = 0.0;
    for (double y : y_true) {
        double diff = y - mean;
        sst += diff * diff;
    }
    
    // Residual sum of squares
    double ssr = 0.0;
    for (size_t i = 0; i < y_true.size(); i++) {
        double diff = y_true[i] - y_pred[i];
        ssr += diff * diff;
    }
    
    return 1.0 - ssr / (sst + 1e-12);
}