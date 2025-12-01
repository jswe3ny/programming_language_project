#include <iostream>
#include <iomanip>
#include <chrono>
#include <stdexcept>
#include "utils/utils.h"
#include "utils/metrics.h"
#include "algos/linear_regression.h"
#include "algos/logistic_regression.h"
#include "algos/knn.h"
#include "algos/decision_tree.h"
#include "algos/naive_bayes.h"

struct CliConfig {
    std::string train_path;
    std::string test_path;
    std::string target_name;
    std::string algo;
    double lr = 0.01;
    int epochs = 100;
    int k = 5;
    int max_depth = 10;
    double l2 = 0.0;
    bool normalize = false;
};

void print_usage() {
    std::cerr << "Usage: program [OPTIONS]\n\n";
    std::cerr << "Required options:\n";
    std::cerr << "  --train <path>       Path to training CSV file\n";
    std::cerr << "  --test <path>        Path to test CSV file\n";
    std::cerr << "  --target <name>      Name of target column\n";
    std::cerr << "  --algo <algorithm>   Algorithm: linear|logistic|knn|tree|nb\n\n";
    std::cerr << "Optional parameters:\n";
    std::cerr << "  --lr <float>         Learning rate (default: 0.01)\n";
    std::cerr << "  --epochs <int>       Number of epochs (default: 100)\n";
    std::cerr << "  --k <int>            Number of neighbors for kNN (default: 5)\n";
    std::cerr << "  --max_depth <int>    Max depth for decision tree (default: 10)\n";
    std::cerr << "  --l2 <float>         L2 regularization (default: 0.0)\n";
    std::cerr << "  --normalize          Apply z-score normalization\n";
}

CliConfig parse_args(int argc, char* argv[]) {
    CliConfig config;
    
    if (argc < 2) {
        print_usage();
        throw std::runtime_error("No arguments provided");
    }
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--train") {
            if (i + 1 >= argc) throw std::runtime_error("--train requires a path argument");
            config.train_path = argv[++i];
        }
        else if (arg == "--test") {
            if (i + 1 >= argc) throw std::runtime_error("--test requires a path argument");
            config.test_path = argv[++i];
        }
        else if (arg == "--target") {
            if (i + 1 >= argc) throw std::runtime_error("--target requires a name argument");
            config.target_name = argv[++i];
        }
        else if (arg == "--algo") {
            if (i + 1 >= argc) throw std::runtime_error("--algo requires an algorithm name");
            config.algo = argv[++i];
            if (config.algo != "linear" && config.algo != "logistic" && 
                config.algo != "knn" && config.algo != "tree" && config.algo != "nb") {
                throw std::runtime_error("Invalid algorithm. Must be: linear|logistic|knn|tree|nb");
            }
        }
        else if (arg == "--lr") {
            if (i + 1 >= argc) throw std::runtime_error("--lr requires a float value");
            config.lr = std::stod(argv[++i]);
            if (config.lr <= 0) throw std::runtime_error("Learning rate must be positive");
        }
        else if (arg == "--epochs") {
            if (i + 1 >= argc) throw std::runtime_error("--epochs requires an integer value");
            config.epochs = std::stoi(argv[++i]);
            if (config.epochs <= 0) throw std::runtime_error("Epochs must be positive");
        }
        else if (arg == "--k") {
            if (i + 1 >= argc) throw std::runtime_error("--k requires an integer value");
            config.k = std::stoi(argv[++i]);
            if (config.k <= 0) throw std::runtime_error("k must be positive");
        }
        else if (arg == "--max_depth") {
            if (i + 1 >= argc) throw std::runtime_error("--max_depth requires an integer value");
            config.max_depth = std::stoi(argv[++i]);
            if (config.max_depth <= 0) throw std::runtime_error("max_depth must be positive");
        }
        else if (arg == "--l2") {
            if (i + 1 >= argc) throw std::runtime_error("--l2 requires a float value");
            config.l2 = std::stod(argv[++i]);
            if (config.l2 < 0) throw std::runtime_error("L2 regularization must be non-negative");
        }
        else if (arg == "--normalize") {
            config.normalize = true;
        }
        else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    if (config.train_path.empty()) throw std::runtime_error("--train is required");
    if (config.test_path.empty()) throw std::runtime_error("--test is required");
    if (config.target_name.empty()) throw std::runtime_error("--target is required");
    if (config.algo.empty()) throw std::runtime_error("--algo is required");
    
    return config;
}

bool is_classification_algo(const std::string& algo) {
    return algo == "logistic" || algo == "knn" || algo == "tree" || algo == "nb";
}

int main(int argc, char* argv[]) {
    std::cout << std::fixed << std::setprecision(4);
    
    try {
        CliConfig config = parse_args(argc, argv);

        Dataset train_data = load_csv(config.train_path, config.target_name);
        if (train_data.X.empty()) {
            throw std::runtime_error("Training data is empty");
        }

        Dataset test_data = load_csv(config.test_path, config.target_name);
        if (test_data.X.empty()) {
            throw std::runtime_error("Test data is empty");
        }
        
        Matrix X_train = train_data.X;
        Vector y_train = train_data.y;
        Matrix X_test = test_data.X;
        Vector y_test = test_data.y;

        if (config.normalize) {
            NormStats stats = zscore_normalize(X_train);
            apply_normalization(X_test, stats);
        }

        auto start_time = std::chrono::high_resolution_clock::now();
        
        Vector predictions;

        if (config.algo == "linear") {
            LinearModel model = linear_regression_fit(X_train, y_train, config.l2);
            predictions = linear_regression_predict(X_test, model);
        }
        else if (config.algo == "logistic") {
            LogisticConfig log_config;
            log_config.learning_rate = config.lr;
            log_config.epochs = config.epochs;
            log_config.l2 = config.l2;
            log_config.verbose = false;
            
            LogisticModel model = logistic_regression_fit(X_train, y_train, log_config);
            predictions = logistic_regression_predict(X_test, model);
        }
        else if (config.algo == "knn") {
            KNNConfig knn_config;
            knn_config.k = config.k;
            knn_config.distance = DistanceMetric::EUCLIDEAN;
            knn_config.weighted = false;
            knn_config.tie_break = TieBreak::SMALLEST_LABEL;
            
            predictions = knn_predict(X_train, y_train, X_test, knn_config);
        }
        else if (config.algo == "tree") {
            DecisionTreeConfig tree_config;
            tree_config.max_depth = config.max_depth;
            tree_config.min_samples_split = 2;
            tree_config.n_bins = 16;
            
            auto tree = decision_tree_fit(X_train, y_train, tree_config);
            predictions = decision_tree_predict(X_test, tree);
        }
        else if (config.algo == "nb") {
            GaussianNBConfig nb_config;
            nb_config.var_smoothing = 1e-9;
            
            GaussianNBModel model = gaussian_nb_fit(X_train, y_train, nb_config);
            predictions = gaussian_nb_predict(X_test, model);
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;

        std::cout << "Algorithm: " << config.algo << "\n";
        std::cout << "Train time: " << elapsed.count() << " seconds\n";
        
        if (is_classification_algo(config.algo)) {
            std::cout << "Test Accuracy: " << accuracy(y_test, predictions) << "\n";
            std::cout << "Macro-F1: " << macro_f1(y_test, predictions) << "\n";
        } else {
            std::cout << "RMSE: " << rmse(y_test, predictions) << "\n";
            std::cout << "R²: " << r2_score(y_test, predictions) << "\n";
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}