#include "decision_tree.h"
#include <cmath>
#include <algorithm>
#include <map>
#include <set>

double entropy(const Vector& y) {
    if (y.empty()) return 0.0;
    
    std::map<double, int> counts;
    for (double label : y) {
        counts[label]++;
    }
    
    double H = 0.0;
    int n = y.size();
    for (const auto& pair : counts) {
        double p = (double)pair.second / n;
        H -= p * std::log2(p + 1e-12);
    }
    
    return H;
}

double information_gain(const Vector& y, const Vector& x_col) {
    double H = entropy(y);
    
    std::map<double, std::vector<double>> groups;
    for (size_t i = 0; i < y.size(); i++) {
        groups[x_col[i]].push_back(y[i]);
    }
    
    double cond_entropy = 0.0;
    for (const auto& pair : groups) {
        double weight = (double)pair.second.size() / y.size();
        cond_entropy += weight * entropy(pair.second);
    }
    
    return H - cond_entropy;
}

std::vector<double> compute_bin_edges(const Vector& data, int n_bins) {
    if (data.empty()) return std::vector<double>();
    
    Vector sorted_data = data;
    std::sort(sorted_data.begin(), sorted_data.end());
    
    double min_val = sorted_data.front();
    double max_val = sorted_data.back();
    double range = max_val - min_val;
    
    if (range < 1e-10) {
        return {min_val, max_val + 1.0};
    }
    
    std::vector<double> edges;
    for (int i = 0; i <= n_bins; i++) {
        edges.push_back(min_val + (range * i) / n_bins);
    }
    
    return edges;
}

Vector digitize(const Vector& data, const std::vector<double>& edges) {
    Vector bins(data.size());
    for (size_t i = 0; i < data.size(); i++) {
        int bin = 0;
        for (size_t j = 0; j < edges.size() - 1; j++) {
            if (data[i] >= edges[j] && data[i] < edges[j + 1]) {
                bin = j;
                break;
            }
        }
        if (data[i] >= edges.back() - 1e-10) {
            bin = edges.size() - 2;
        }
        bins[i] = bin;
    }
    return bins;
}

struct BestSplit {
    int feature = -1;
    std::vector<double> bin_edges;
    double gain = -1.0;
};

BestSplit find_best_split(const Matrix& X, const Vector& y, int n_bins) {
    BestSplit best;
    int n = X.size();
    int d = X[0].size();
    
    for (int j = 0; j < d; j++) {
        // Extract column
        Vector col(n);
        for (int i = 0; i < n; i++) {
            col[i] = X[i][j];
        }
        
        // Compute bin edges
        std::vector<double> edges = compute_bin_edges(col, n_bins);
        Vector bins = digitize(col, edges);
        
        // Compute information gain
        double gain = information_gain(y, bins);
        
        if (gain > best.gain) {
            best.gain = gain;
            best.feature = j;
            best.bin_edges = edges;
        }
    }
    
    return best;
}

double majority_label(const Vector& y) {
    if (y.empty()) return 0.0;
    
    std::map<double, int> counts;
    for (double label : y) {
        counts[label]++;
    }
    
    double best_label = 0.0;
    int best_count = 0;
    for (const auto& pair : counts) {
        if (pair.second > best_count) {
            best_count = pair.second;
            best_label = pair.first;
        }
    }
    
    return best_label;
}

std::shared_ptr<TreeNode> decision_tree_fit_recursive(
    const Matrix& X, const Vector& y, 
    int depth, const DecisionTreeConfig& config) {
    
    auto node = std::make_shared<TreeNode>();
    
    // Stopping conditions
    std::set<double> unique_labels(y.begin(), y.end());
    if (depth >= config.max_depth || 
        unique_labels.size() == 1 || 
        (int)y.size() < config.min_samples_split) {
        node->is_leaf = true;
        node->label = majority_label(y);
        return node;
    }
    
    // Find best split
    BestSplit split = find_best_split(X, y, config.n_bins);
    
    if (split.feature == -1 || split.bin_edges.empty()) {
        node->is_leaf = true;
        node->label = majority_label(y);
        return node;
    }
    
    // Extract feature column and discretize
    int n = X.size();
    Vector col(n);
    for (int i = 0; i < n; i++) {
        col[i] = X[split.feature][i];
    }
    Vector bins = digitize(col, split.bin_edges);
    
    // Partition data by bin
    std::map<int, std::pair<Matrix, Vector>> partitions;
    for (int i = 0; i < n; i++) {
        int bin = (int)bins[i];
        partitions[bin].first.push_back(X[i]);
        partitions[bin].second.push_back(y[i]);
    }
    
    // Build children recursively
    node->is_leaf = false;
    node->feature = split.feature;
    node->bin_edges = split.bin_edges;
    
    for (const auto& pair : partitions) {
        int bin = pair.first;
        const Matrix& X_sub = pair.second.first;
        const Vector& y_sub = pair.second.second;
        
        node->children[bin] = decision_tree_fit_recursive(
            X_sub, y_sub, depth + 1, config);
    }
    
    return node;
}

std::shared_ptr<TreeNode> decision_tree_fit(const Matrix& X, const Vector& y,
                                           const DecisionTreeConfig& config) {
    return decision_tree_fit_recursive(X, y, 0, config);
}

double predict_one(const Vector& x, const std::shared_ptr<TreeNode>& node) {
    if (node->is_leaf) {
        return node->label;
    }
    
    // Discretize feature value
    double val = x[node->feature];
    int bin = 0;
    for (size_t i = 0; i < node->bin_edges.size() - 1; i++) {
        if (val >= node->bin_edges[i] && val < node->bin_edges[i + 1]) {
            bin = i;
            break;
        }
    }
    if (val >= node->bin_edges.back() - 1e-10) {
        bin = node->bin_edges.size() - 2;
    }
    
    // Follow appropriate child
    auto it = node->children.find(bin);
    if (it == node->children.end()) {
        // Fallback if bin not seen during training
        return node->children.begin()->second->label;
    }
    
    return predict_one(x, it->second);
}

Vector decision_tree_predict(const Matrix& X, const std::shared_ptr<TreeNode>& tree) {
    Vector predictions(X.size());
    for (size_t i = 0; i < X.size(); i++) {
        predictions[i] = predict_one(X[i], tree);
    }
    return predictions;
}