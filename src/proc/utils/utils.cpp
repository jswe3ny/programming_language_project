#include "utils.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <random>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <unordered_map>

// Helper: trim whitespace
std::string trim(const std::string& str) {
    size_t first = str.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return "";
    size_t last = str.find_last_not_of(" \t\r\n");
    return str.substr(first, last - first + 1);
}

// Helper: parse CSV line
std::vector<std::string> parse_csv_line(const std::string& line) {
    std::vector<std::string> result;
    std::string cell;
    bool in_quotes = false;
    
    for (char c : line) {
        if (c == '"') {
            in_quotes = !in_quotes;
        } else if (c == ',' && !in_quotes) {
            result.push_back(trim(cell));
            cell.clear();
        } else {
            cell += c;
        }
    }
    result.push_back(trim(cell));
    return result;
}

Dataset load_csv(const std::string& path, const std::string& target_col) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + path);
    }

    Dataset ds;
    std::string line;
    std::vector<std::string> headers;

    // Read header
    if (std::getline(file, line)) {
        headers = parse_csv_line(line);
    }

    // Find target column
    int target_idx = -1;
    for (size_t i = 0; i < headers.size(); i++) {
        if (headers[i] == target_col) {
            target_idx = static_cast<int>(i);
            break;
        }
    }
    if (target_idx == -1) {
        throw std::runtime_error("Target column not found: " + target_col);
    }

    // Store feature names (excluding target)
    for (size_t i = 0; i < headers.size(); i++) {
        if (static_cast<int>(i) != target_idx) {
            ds.feature_names.push_back(headers[i]);
        }
    }

    // Read rows (features as strings, target as strings)
    std::vector<std::vector<std::string>> raw_data;
    std::vector<std::string> target_data;

    while (std::getline(file, line)) {
        auto values = parse_csv_line(line);
        if (values.size() != headers.size()) continue;

        std::vector<std::string> row;
        row.reserve(headers.size() - 1);

        for (size_t i = 0; i < values.size(); i++) {
            if (static_cast<int>(i) == target_idx) {
                target_data.push_back(values[i]);
            } else {
                row.push_back(values[i]);
            }
        }
        raw_data.push_back(std::move(row));
    }

    // Detect which feature columns are numeric
    const int n_features = static_cast<int>(ds.feature_names.size());
    std::vector<bool> is_numeric(n_features, true);

    auto is_fully_numeric = [](const std::string& s) -> bool {
        try {
            size_t pos = 0;
            (void)std::stod(s, &pos);
            return pos == s.size();
        } catch (...) { return false; }
    };

    for (int j = 0; j < n_features; j++) {
        for (const auto& row : raw_data) {
            if (!is_fully_numeric(row[j])) {
                is_numeric[j] = false;
                break;
            }
        }
    }

    // Convert features: numeric -> double, non-numeric -> 0.0 (placeholder)
    // (Keeps current behavior for features; change to one-hot later if desired.)
    ds.X.resize(raw_data.size());
    for (size_t i = 0; i < raw_data.size(); i++) {
        ds.X[i].resize(n_features);
        for (int j = 0; j < n_features; j++) {
            if (is_numeric[j]) {
                try {
                    ds.X[i][j] = std::stod(raw_data[i][j]);
                } catch (...) {
                    ds.X[i][j] = 0.0;
                }
            } else {
                ds.X[i][j] = 0.0; // placeholder for categorical (consistent with your current code)
            }
        }
    }

    // ----- FIXED TARGET PARSING -----
    // Decide if the target column is numeric across the dataset
    bool target_is_numeric = true;
    for (const auto& s : target_data) {
        if (!is_fully_numeric(s)) {
            target_is_numeric = false;
            break;
        }
    }

    ds.y.resize(target_data.size());

    if (target_is_numeric) {
        // Numeric target (e.g., "hours.per.week")
        for (size_t i = 0; i < target_data.size(); i++) {
            try {
                ds.y[i] = std::stod(target_data[i]);
            } catch (...) {
                ds.y[i] = 0.0;
            }
        }
    } else {
        // Categorical target. First try your Adult Income mapping.
        // If it doesn't match that pattern, fall back to a generic label map.
        // (This keeps things robust for other datasets.)
        auto lower_nospace_nodot = [](std::string t) {
            std::transform(t.begin(), t.end(), t.begin(), ::tolower);
            t.erase(std::remove(t.begin(), t.end(), '.'), t.end());
            t.erase(std::remove(t.begin(), t.end(), ' '), t.end());
            return t;
        };

        bool looks_like_adult_income = true;
        for (const auto& s : target_data) {
            auto t = lower_nospace_nodot(s);
            if (!(t == "<=50k" || t == ">50k")) {
                looks_like_adult_income = false;
                break;
            }
        }

        if (looks_like_adult_income) {
            // Uses your existing helper
            ds.y = map_income_to_binary(target_data);
        } else {
            // Generic categorical mapping to 0..K-1
            std::unordered_map<std::string, int> labmap;
            int next = 0;
            for (size_t i = 0; i < target_data.size(); i++) {
                std::string key = lower_nospace_nodot(target_data[i]);
                if (!labmap.count(key)) labmap[key] = next++;
                ds.y[i] = static_cast<double>(labmap[key]);
            }
        }
    }

    return ds;
}

Vector map_income_to_binary(const std::vector<std::string>& labels) {
    Vector result(labels.size());
    for (size_t i = 0; i < labels.size(); i++) {
        std::string low = labels[i];
        std::transform(low.begin(), low.end(), low.begin(), ::tolower);
        low.erase(std::remove(low.begin(), low.end(), '.'), low.end());
        low.erase(std::remove(low.begin(), low.end(), ' '), low.end());
        
        if (low.find("<=50k") != std::string::npos || low == "<=50k") {
            result[i] = 0.0;
        } else {
            result[i] = 1.0;
        }
    }
    return result;
}

NormStats zscore_normalize(Matrix& X, const NormStats* existing) {
    if (X.empty()) return NormStats();
    
    int n = X.size();
    int d = X[0].size();
    
    NormStats stats;
    stats.means.resize(d, 0.0);
    stats.stds.resize(d, 1.0);
    
    for (int j = 0; j < d; j++) {
        stats.numeric_indices.push_back(j);
    }
    
    if (existing) {
        stats.means = existing->means;
        stats.stds = existing->stds;
    } else {
        // Compute means
        for (int j = 0; j < d; j++) {
            double sum = 0.0;
            for (int i = 0; i < n; i++) {
                sum += X[i][j];
            }
            stats.means[j] = sum / n;
        }
        
        // Compute stds
        for (int j = 0; j < d; j++) {
            double sum_sq = 0.0;
            for (int i = 0; i < n; i++) {
                double diff = X[i][j] - stats.means[j];
                sum_sq += diff * diff;
            }
            stats.stds[j] = std::sqrt(sum_sq / n);
            if (stats.stds[j] < 1e-8) stats.stds[j] = 1.0;
        }
    }
    
    // Apply normalization
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < d; j++) {
            X[i][j] = (X[i][j] - stats.means[j]) / stats.stds[j];
        }
    }
    
    return stats;
}

void apply_normalization(Matrix& X, const NormStats& stats) {
    for (size_t i = 0; i < X.size(); i++) {
        for (size_t j = 0; j < stats.means.size() && j < X[i].size(); j++) {
            X[i][j] = (X[i][j] - stats.means[j]) / stats.stds[j];
        }
    }
}

std::pair<Matrix, Matrix> align_columns(const Matrix& train, Matrix& test, int train_cols) {
    int test_rows = test.size();
    Matrix aligned_test(test_rows, Vector(train_cols, 0.0));
    
    for (int i = 0; i < test_rows; i++) {
        for (int j = 0; j < train_cols && j < (int)test[i].size(); j++) {
            aligned_test[i][j] = test[i][j];
        }
    }
    
    return {train, aligned_test};
}

TrainTestSplit train_test_split(const Matrix& X, const Vector& y, double test_size, int seed) {
    int n = X.size();
    std::vector<int> indices(n);
    for (int i = 0; i < n; i++) indices[i] = i;
    
    std::mt19937 rng(seed);
    std::shuffle(indices.begin(), indices.end(), rng);
    
    int split = (int)((1.0 - test_size) * n);
    
    TrainTestSplit result;
    result.X_train.reserve(split);
    result.X_test.reserve(n - split);
    result.y_train.reserve(split);
    result.y_test.reserve(n - split);
    
    for (int i = 0; i < split; i++) {
        result.X_train.push_back(X[indices[i]]);
        result.y_train.push_back(y[indices[i]]);
    }
    
    for (int i = split; i < n; i++) {
        result.X_test.push_back(X[indices[i]]);
        result.y_test.push_back(y[indices[i]]);
    }
    
    return result;
}

void print_matrix_shape(const Matrix& X, const std::string& name) {
    if (X.empty()) {
        std::cout << name << ": (0, 0)" << std::endl;
    } else {
        std::cout << name << ": (" << X.size() << ", " << X[0].size() << ")" << std::endl;
    }
}

double vector_mean(const Vector& v) {
    if (v.empty()) return 0.0;
    double sum = 0.0;
    for (double x : v) sum += x;
    return sum / v.size();
}

double vector_std(const Vector& v) {
    if (v.empty()) return 0.0;
    double mean = vector_mean(v);
    double sum_sq = 0.0;
    for (double x : v) {
        double diff = x - mean;
        sum_sq += diff * diff;
    }
    return std::sqrt(sum_sq / v.size());
}