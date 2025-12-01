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

    std::string line;
    std::vector<std::string> headers;

    if (std::getline(file, line)) {
        headers = parse_csv_line(line);
    }

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

    std::vector<std::vector<std::string>> raw_data;
    std::vector<std::string> target_data;

    while (std::getline(file, line)) {
        auto values = parse_csv_line(line);
        if (values.size() != headers.size()) continue;

        std::vector<std::string> row;
        for (size_t i = 0; i < values.size(); i++) {
            if (static_cast<int>(i) == target_idx) {
                target_data.push_back(values[i]);
            } else {
                row.push_back(values[i]);
            }
        }
        raw_data.push_back(std::move(row));
    }

    // Detect numeric vs categorical columns
    const int n_features = static_cast<int>(headers.size()) - 1;
    std::vector<bool> is_numeric(n_features, true);
    std::vector<std::set<std::string>> unique_values(n_features);

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
            }
            unique_values[j].insert(row[j]);
        }
    }

    // one-hot encode
    std::vector<std::string> encoded_feature_names;
    std::vector<std::map<std::string, int>> category_maps(n_features);

    int total_features = 0;
    for (int j = 0; j < n_features; j++) {
        if (is_numeric[j]) {
            encoded_feature_names.push_back(headers[j < target_idx ? j : j + 1]);
            total_features++;
        } else {
            int col_idx = 0;
            for (const auto& val : unique_values[j]) {
                std::string feat_name = headers[j < target_idx ? j : j + 1] + "_" + val;
                encoded_feature_names.push_back(feat_name);
                category_maps[j][val] = col_idx++;
                total_features++;
            }
        }
    }

    Dataset ds;
    ds.feature_names = encoded_feature_names;
    ds.X.resize(raw_data.size(), Vector(total_features, 0.0));

    for (size_t i = 0; i < raw_data.size(); i++) {
        int out_col = 0;
        for (int j = 0; j < n_features; j++) {
            if (is_numeric[j]) {
                try {
                    ds.X[i][out_col++] = std::stod(raw_data[i][j]);
                } catch (...) {
                    ds.X[i][out_col++] = 0.0;
                }
            } else {
                int cat_col = category_maps[j][raw_data[i][j]];
                int num_categories = unique_values[j].size();
                for (int k = 0; k < num_categories; k++) {
                    ds.X[i][out_col + k] = (k == cat_col) ? 1.0 : 0.0;
                }
                out_col += num_categories;
            }
        }
    }

    bool target_is_numeric = true;
    for (const auto& s : target_data) {
        if (!is_fully_numeric(s)) {
            target_is_numeric = false;
            break;
        }
    }

    ds.y.resize(target_data.size());
    if (target_is_numeric) {
        for (size_t i = 0; i < target_data.size(); i++) {
            try {
                ds.y[i] = std::stod(target_data[i]);
            } catch (...) {
                ds.y[i] = 0.0;
            }
        }
    } else {
        ds.y = map_income_to_binary(target_data);
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
        for (int j = 0; j < d; j++) {
            double sum = 0.0;
            for (int i = 0; i < n; i++) {
                sum += X[i][j];
            }
            stats.means[j] = sum / n;
        }
        
        // Compute stds (SAMPLE standard deviation to match pandas default)
        for (int j = 0; j < d; j++) {
            double sum_sq = 0.0;
            for (int i = 0; i < n; i++) {
                double diff = X[i][j] - stats.means[j];
                sum_sq += diff * diff;
            }
            // Use n-1 for sample std (ddof=1) to match pandas
            if (n > 1) {
                stats.stds[j] = std::sqrt(sum_sq / (n - 1));
            } else {
                stats.stds[j] = 1.0;
            }
            if (stats.stds[j] < 1e-8) stats.stds[j] = 1.0;
        }
    }

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