package ml.models;

import ml.core.Dataset;
import ml.core.Model;

import java.util.*;


public class DecisionTree implements Model {

    private final int maxDepth;
    private final int minSamples;
    private final int nBins;

    private Node root;
    private double[][] binEdges;   // [feature][bin boundaries]

    public DecisionTree(int maxDepth, int minSamples, int nBins) {
        this.maxDepth = maxDepth;
        this.minSamples = minSamples;
        this.nBins = nBins;
    }

    @Override
    public String name() { return "DecisionTree(ID3)"; }

    @Override
    public void fit(Dataset train) {
        int N = train.X.length;
        int D = train.X[0].length;

        // Precompute bin edges for each feature (equal-width bins)
        binEdges = new double[D][nBins + 1];
        for (int j = 0; j < D; j++) {
            double min = Double.POSITIVE_INFINITY;
            double max = Double.NEGATIVE_INFINITY;
            for (int i = 0; i < N; i++) {
                double v = train.X[i][j];
                if (v < min) min = v;
                if (v > max) max = v;
            }
            if (min == max) {
                Arrays.fill(binEdges[j], min);
            } else {
                double step = (max - min) / nBins;
                for (int b = 0; b <= nBins; b++)
                    binEdges[j][b] = min + b * step;
                binEdges[j][nBins] = max; // ensure last boundary exactly max
            }
        }

        // Convert all feature values into binned values
        int[][] Xbin = binAll(train.X);

        int[] rows = new int[N];
        for (int i = 0; i < N; i++) rows[i] = i;

        this.root = buildTree(Xbin, train.y, rows, 0);
    }

    // ================================================================
    // Prediction
    // ================================================================

    @Override
    public double[] predict(double[][] X) {
        int[][] Xb = binAll(X);
        double[] out = new double[X.length];
        for (int i = 0; i < X.length; i++)
            out[i] = predictOne(Xb[i], root);
        return out;
    }

    private double predictOne(int[] xb, Node node) {
        while (!node.isLeaf) {
            int feature = node.feature;
            int b = xb[feature];
            if (b < node.children.length)
                node = node.children[b];
            else
                node = node.children[node.children.length - 1];
        }
        return node.prediction;
    }

    // ================================================================
    // Tree Building
    // ================================================================

    private Node buildTree(int[][] X, double[] y, int[] rows, int depth) {

        // Count class frequencies
        int count0 = 0, count1 = 0;
        for (int r : rows) {
            if (y[r] == 1.0) count1++;
            else count0++;
        }

        // Majority vote
        double majority = (count1 >= count0) ? 1.0 : 0.0;

        // Leaf conditions
        if (depth >= maxDepth || rows.length < minSamples) {
            return new Node(true, majority);
        }
        if (count0 == 0 || count1 == 0) {
            return new Node(true, majority); // pure leaf
        }

        // Find best split
        int bestFeature = -1;
        double bestGain = -1;

        int D = X[0].length;

        for (int j = 0; j < D; j++) {
            double gain = informationGain(X, y, rows, j);
            if (gain > bestGain) {
                bestGain = gain;
                bestFeature = j;
            }
        }

        if (bestFeature == -1 || bestGain <= 0.0) {
            return new Node(true, majority);
        }

        // Split rows by best feature's bin index
        Map<Integer, List<Integer>> bins = new HashMap<>();
        for (int r : rows) {
            int b = X[r][bestFeature];
            bins.computeIfAbsent(b, k -> new ArrayList<>()).add(r);
        }

        // Create node
        Node node = new Node(false, majority);
        node.feature = bestFeature;

        // Maximum possible bins = nBins
        node.children = new Node[nBins];

        for (int b = 0; b < nBins; b++) {
            List<Integer> group = bins.getOrDefault(b, Collections.emptyList());
            if (group.isEmpty()) {
                node.children[b] = new Node(true, majority);
            } else {
                int[] subrows = group.stream().mapToInt(i -> i).toArray();
                node.children[b] = buildTree(X, y, subrows, depth + 1);
            }
        }

        return node;
    }

    // ================================================================
    // Information Gain
    // ================================================================

    private double informationGain(int[][] X, double[] y, int[] rows, int feature) {
        double parentEntropy = entropy(y, rows);

        // Partition child groups
        Map<Integer, List<Integer>> bins = new HashMap<>();
        for (int r : rows) {
            int b = X[r][feature];
            bins.computeIfAbsent(b, k -> new ArrayList<>()).add(r);
        }

        double weightedEntropy = 0;
        int N = rows.length;

        for (java.util.Map.Entry<Integer, List<Integer>> entry : bins.entrySet()) {    
            int[] sub = entry.getValue().stream().mapToInt(i -> i).toArray();
            weightedEntropy += (sub.length / (double) N) * entropy(y, sub);
        }

        return parentEntropy - weightedEntropy;
    }

    private double entropy(double[] y, int[] rows) {
        if (rows.length == 0) return 0.0;

        double count0 = 0, count1 = 0;
        for (int r : rows) {
            if (y[r] == 1.0) count1++;
            else count0++;
        }

        double p0 = count0 / rows.length;
        double p1 = count1 / rows.length;

        double h = 0;
        if (p0 > 0) h -= p0 * Math.log(p0) / Math.log(2);
        if (p1 > 0) h -= p1 * Math.log(p1) / Math.log(2);
        return h;
    }

    // ================================================================
    // Binning
    // ================================================================

    private int[][] binAll(double[][] X) {
        int N = X.length, D = X[0].length;
        int[][] out = new int[N][D];

        for (int j = 0; j < D; j++) {
            double[] edges = binEdges[j];

            for (int i = 0; i < N; i++) {
                double v = X[i][j];

                int b = 0;
                while (b < nBins && v > edges[b + 1]) b++;

                if (b >= nBins) b = nBins - 1;
                out[i][j] = b;
            }
        }

        return out;
    }

    // ================================================================
    // Node class
    // ================================================================

    private static class Node {
        boolean isLeaf;
        double prediction; // used only if leaf
        int feature;       // index of split feature
        Node[] children;   // child nodes for each bin

        Node(boolean isLeaf, double pred) {
            this.isLeaf = isLeaf;
            this.prediction = pred;
        }
    }
}
