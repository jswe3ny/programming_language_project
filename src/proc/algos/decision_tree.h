#ifndef DECISION_TREE_H
#define DECISION_TREE_H

#include <vector>
#include <map>
#include <memory>

using Matrix = std::vector<std::vector<double>>;
using Vector = std::vector<double>;

struct TreeNode {
    bool is_leaf = false;
    double label = 0.0;             // For leaf nodes
    int feature = -1;               // For internal nodes
    std::vector<double> bin_edges;  // For discretization
    std::map<int, std::shared_ptr<TreeNode>> children;
};

struct DecisionTreeConfig {
    int max_depth = 5;
    int min_samples_split = 5;
    int n_bins = 12;
};

// Build decision tree using ID3 algorithm
std::shared_ptr<TreeNode> decision_tree_fit(const Matrix& X, const Vector& y,
                                           const DecisionTreeConfig& config = DecisionTreeConfig());

// Predict using decision tree
Vector decision_tree_predict(const Matrix& X, const std::shared_ptr<TreeNode>& tree);

// Helper functions
double entropy(const Vector& y);
double information_gain(const Vector& y, const Vector& x_col);

#endif // DECISION_TREE_H