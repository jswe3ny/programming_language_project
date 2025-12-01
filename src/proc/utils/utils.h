#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <string>
#include <map>
#include <set>

using Matrix = std::vector<std::vector<double>>;
using Vector = std::vector<double>;

struct Dataset {
    Matrix X;
    Vector y;
    std::vector<std::string> feature_names;
    std::vector<std::string> categorical_features;
};

struct NormStats {
    Vector means;
    Vector stds;
    std::vector<int> numeric_indices;
};

Dataset load_csv(const std::string& path, const std::string& target_col);

NormStats zscore_normalize(Matrix& X, const NormStats* existing = nullptr);

void apply_normalization(Matrix& X, const NormStats& stats);

std::pair<Matrix, Matrix> align_columns(const Matrix& train, Matrix& test,
                                        int train_cols);

struct TrainTestSplit {
    Matrix X_train, X_test;
    Vector y_train, y_test;
};

TrainTestSplit train_test_split(const Matrix& X, const Vector& y,
                                double test_size = 0.2, int seed = 42);

Vector map_income_to_binary(const std::vector<std::string>& labels);
void print_matrix_shape(const Matrix& X, const std::string& name = "Matrix");
double vector_mean(const Vector& v);
double vector_std(const Vector& v);

#endif