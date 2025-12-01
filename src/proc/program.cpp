#include <iostream>
#include <iomanip>
#include "utils/utils.h"
#include "utils/metrics.h"
#include "algos/linear_regression.h"
#include "algos/logistic_regression.h"
#include "algos/knn.h"
#include "algos/decision_tree.h"
#include "algos/naive_bayes.h"

void print_header(const std::string& title) {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << title << "\n";
    std::cout << std::string(60, '=') << "\n";
}

int main() {
    const std::string csv_path = "../data/adult_income_cleaned.csv";
    const std::string target_col_cls = "income";
    const std::string target_col_reg = "hours.per.week";
    const int RANDOM_SEED = 42;

    std::cout << std::fixed << std::setprecision(4);

    try {
        // =====================================================
        // CLASSIFICATION TASK
        // =====================================================
        print_header("LOADING DATA FOR CLASSIFICATION");

        Dataset data_cls = load_csv(csv_path, target_col_cls);
        std::cout << "Loaded " << data_cls.X.size() << " samples with "
                  << data_cls.X[0].size() << " features\n";

        // Split data
        auto split_cls = train_test_split(data_cls.X, data_cls.y, 0.3, RANDOM_SEED);

        // Normalize
        NormStats stats_cls = zscore_normalize(split_cls.X_train);
        apply_normalization(split_cls.X_test, stats_cls);

        std::cout << "Training set: " << split_cls.X_train.size() << " samples\n";
        std::cout << "Test set: " << split_cls.X_test.size() << " samples\n";

        // =====================================================
        // LOGISTIC REGRESSION
        // =====================================================
        print_header("LOGISTIC REGRESSION");

        LogisticConfig log_config;
        log_config.learning_rate = 0.1;
        log_config.epochs = 300;
        log_config.l2 = 1e-3;
        log_config.verbose = true;

        LogisticModel log_model = logistic_regression_fit(
            split_cls.X_train, split_cls.y_train, log_config);

        Vector pred_log = logistic_regression_predict(split_cls.X_test, log_model);

        std::cout << "\nLogistic Regression Results:\n";
        std::cout << "  Accuracy:  " << accuracy(split_cls.y_test, pred_log) << "\n";
        std::cout << "  Macro-F1:  " << macro_f1(split_cls.y_test, pred_log) << "\n";

        // =====================================================
        // K-NEAREST NEIGHBORS
        // =====================================================
        print_header("K-NEAREST NEIGHBORS");

        KNNConfig knn_config;
        knn_config.k = 7;
        knn_config.distance = DistanceMetric::EUCLIDEAN;
        knn_config.weighted = false;
        knn_config.tie_break = TieBreak::SMALLEST_LABEL;

        Vector pred_knn = knn_predict(
            split_cls.X_train, split_cls.y_train,
            split_cls.X_test, knn_config);

        std::cout << "kNN Results (k=" << knn_config.k << "):\n";
        std::cout << "  Accuracy:  " << accuracy(split_cls.y_test, pred_knn) << "\n";
        std::cout << "  Macro-F1:  " << macro_f1(split_cls.y_test, pred_knn) << "\n";

        // =====================================================
        // DECISION TREE (ID3)
        // =====================================================
        print_header("DECISION TREE (ID3)");

        DecisionTreeConfig tree_config;
        tree_config.max_depth = 5;
        tree_config.min_samples_split = 10;
        tree_config.n_bins = 16;

        auto tree = decision_tree_fit(
            split_cls.X_train, split_cls.y_train, tree_config);

        Vector pred_tree = decision_tree_predict(split_cls.X_test, tree);

        std::cout << "Decision Tree Results:\n";
        std::cout << "  Accuracy:  " << accuracy(split_cls.y_test, pred_tree) << "\n";
        std::cout << "  Macro-F1:  " << macro_f1(split_cls.y_test, pred_tree) << "\n";

        // =====================================================
        // GAUSSIAN NAIVE BAYES
        // =====================================================
        print_header("GAUSSIAN NAIVE BAYES");

        GaussianNBConfig gnb_config;
        gnb_config.var_smoothing = 1e-9;

        GaussianNBModel gnb_model = gaussian_nb_fit(
            split_cls.X_train, split_cls.y_train, gnb_config);

        Vector pred_gnb = gaussian_nb_predict(split_cls.X_test, gnb_model);

        std::cout << "Gaussian Naive Bayes Results:\n";
        std::cout << "  Accuracy:  " << accuracy(split_cls.y_test, pred_gnb) << "\n";
        std::cout << "  Macro-F1:  " << macro_f1(split_cls.y_test, pred_gnb) << "\n";

        // =====================================================
        // REGRESSION TASK
        // =====================================================
        print_header("LOADING DATA FOR REGRESSION");

        Dataset data_reg = load_csv(csv_path, target_col_reg);
        std::cout << "Loaded " << data_reg.X.size() << " samples for regression\n";

        // Split data
        auto split_reg = train_test_split(data_reg.X, data_reg.y, 0.3, RANDOM_SEED);

        // Normalize
        NormStats stats_reg = zscore_normalize(split_reg.X_train);
        apply_normalization(split_reg.X_test, stats_reg);

        // =====================================================
        // LINEAR REGRESSION
        // =====================================================
        print_header("LINEAR REGRESSION");

        LinearModel lin_model = linear_regression_fit(
            split_reg.X_train, split_reg.y_train, 0.0);  // l2=0.0

        Vector pred_lin = linear_regression_predict(split_reg.X_test, lin_model);

        std::cout << "Linear Regression Results:\n";
        std::cout << "  RMSE:  " << rmse(split_reg.y_test, pred_lin) << "\n";
        std::cout << "  R²:    " << r2_score(split_reg.y_test, pred_lin) << "\n";

        // =====================================================
        // SUMMARY
        // =====================================================
        print_header("SUMMARY");

        std::cout << "\nClassification Models:\n";
        std::cout << "  Logistic Regression:  Acc=" << accuracy(split_cls.y_test, pred_log)
                  << "  F1=" << macro_f1(split_cls.y_test, pred_log) << "\n";
        std::cout << "  kNN (k=7):            Acc=" << accuracy(split_cls.y_test, pred_knn)
                  << "  F1=" << macro_f1(split_cls.y_test, pred_knn) << "\n";
        std::cout << "  Decision Tree (ID3):  Acc=" << accuracy(split_cls.y_test, pred_tree)
                  << "  F1=" << macro_f1(split_cls.y_test, pred_tree) << "\n";
        std::cout << "  Gaussian Naive Bayes: Acc=" << accuracy(split_cls.y_test, pred_gnb)
                  << "  F1=" << macro_f1(split_cls.y_test, pred_gnb) << "\n";

        std::cout << "\nRegression Model:\n";
        std::cout << "  Linear Regression:    RMSE=" << rmse(split_reg.y_test, pred_lin)
                  << "  R²=" << r2_score(split_reg.y_test, pred_lin) << "\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}