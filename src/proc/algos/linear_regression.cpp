#include "linear_regression.h"
#include <cmath>
#include <stdexcept>

// Matrix multiplication C = A * B
Matrix mat_mult(const Matrix& A, const Matrix& B) {
    int m = A.size();
    int n = B[0].size();
    int k = B.size();
    
    Matrix C(m, Vector(n, 0.0));
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            for (int p = 0; p < k; p++) {
                C[i][j] += A[i][p] * B[p][j];
            }
        }
    }
    return C;
}

// Matrix transpose
Matrix mat_transpose(const Matrix& A) {
    if (A.empty()) return Matrix();
    int m = A.size();
    int n = A[0].size();
    Matrix At(n, Vector(m));
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            At[j][i] = A[i][j];
        }
    }
    return At;
}

// Matrix inverse using Gauss-Jordan elimination (from google)
Matrix mat_inverse(Matrix A) {
    int n = A.size();
    Matrix I(n, Vector(n, 0.0));
    for (int i = 0; i < n; i++) I[i][i] = 1.0;

    for (int i = 0; i < n; i++) {
        double max_val = std::abs(A[i][i]);
        int max_row = i;
        for (int k = i + 1; k < n; k++) {
            if (std::abs(A[k][i]) > max_val) {
                max_val = std::abs(A[k][i]);
                max_row = k;
            }
        }

        if (max_row != i) {
            std::swap(A[i], A[max_row]);
            std::swap(I[i], I[max_row]);
        }

        double pivot = A[i][i];
        if (std::abs(pivot) < 1e-10) pivot = 1e-10;
        
        for (int j = 0; j < n; j++) {
            A[i][j] /= pivot;
            I[i][j] /= pivot;
        }

        for (int k = 0; k < n; k++) {
            if (k != i) {
                double factor = A[k][i];
                for (int j = 0; j < n; j++) {
                    A[k][j] -= factor * A[i][j];
                    I[k][j] -= factor * I[i][j];
                }
            }
        }
    }
    
    return I;
}

LinearModel linear_regression_fit(const Matrix& X, const Vector& y, double l2) {
    int n = X.size();
    int d = X[0].size();

    Matrix X_aug(n, Vector(d + 1));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < d; j++) {
            X_aug[i][j] = X[i][j];
        }
        X_aug[i][d] = 1.0;
    }
    
    // Compute X^T * X
    Matrix Xt = mat_transpose(X_aug);
    Matrix XtX = mat_mult(Xt, X_aug);

    for (int i = 0; i < d; i++) {
        XtX[i][i] += l2;
    }
    
    // Compute (X^T * X)^(-1)
    Matrix XtX_inv = mat_inverse(XtX);
    
    // Compute X^T * y
    Vector Xty(d + 1, 0.0);
    for (int j = 0; j < d + 1; j++) {
        for (int i = 0; i < n; i++) {
            Xty[j] += Xt[j][i] * y[i];
        }
    }
    
    // Compute w = (X^T * X)^(-1) * X^T * y
    Vector w_aug(d + 1, 0.0);
    for (int i = 0; i < d + 1; i++) {
        for (int j = 0; j < d + 1; j++) {
            w_aug[i] += XtX_inv[i][j] * Xty[j];
        }
    }

    LinearModel model;
    model.weights.resize(d);
    for (int i = 0; i < d; i++) {
        model.weights[i] = w_aug[i];
    }
    model.bias = w_aug[d];
    
    return model;
}

Vector linear_regression_predict(const Matrix& X, const LinearModel& model) {
    int n = X.size();
    Vector predictions(n, 0.0);
    
    for (int i = 0; i < n; i++) {
        predictions[i] = model.bias;
        for (size_t j = 0; j < model.weights.size() && j < X[i].size(); j++) {
            predictions[i] += X[i][j] * model.weights[j];
        }
    }
    
    return predictions;
}