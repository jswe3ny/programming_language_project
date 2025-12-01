package ml.models;

import ml.core.Dataset;
import ml.core.Model;

import java.util.*;

/** ID3 Decision Tree (classification) with numeric features via binning. */
public class DecisionTree implements Model {

    public static class Config {
        public final int maxDepth, minSamples, nBins;
        public Config(int maxDepth, int minSamples, int nBins){
            this.maxDepth = Math.max(1, maxDepth);
            this.minSamples = Math.max(2, minSamples);
            this.nBins = Math.max(2, nBins);
        }
    }

    private final Config cfg;
    private Node root;

    public DecisionTree(int maxDepth, int minSamples, int nBins){
        this.cfg = new Config(maxDepth, minSamples, nBins);
    }

    @Override public String name(){
        return "DecisionTree(maxDepth="+cfg.maxDepth+", minSamples="+cfg.minSamples+", bins="+cfg.nBins+")";
    }

    @Override public void fit(Dataset d){
        int n = d.nRows(), p = d.nCols();
        int[] idx = new int[n]; for(int i=0;i<n;i++) idx[i]=i;
        root = build(d.X, d.y, idx, 0, p);
    }

    @Override public double[] predict(double[][] X){
        double[] out = new double[X.length];
        for (int i=0;i<X.length;i++) out[i] = predictOne(root, X[i]);
        return out;
    }

    @Override public boolean isClassifier(){ return true; }

    // ---------- Tree internals ----------

    private static class Node {
        boolean leaf;
        double label;              // if leaf
        int feature;               // if internal
        double[] edges;            // bin edges for the chosen feature
        Map<Integer, Node> kids;   // bin -> child
    }

    private Node build(double[][] X, double[] y, int[] rows, int depth, int p){
        Node node = new Node();

        // stopping conditions
        double maj = majority(y, rows);
        if (depth >= cfg.maxDepth || rows.length < cfg.minSamples || pure(y, rows)){
            node.leaf = true; node.label = maj; return node;
        }

        // find best split among features
        Best best = bestSplit(X, y, rows, p, cfg.nBins);
        if (best == null || best.gain <= 1e-12){
            node.leaf = true; node.label = maj; return node;
        }

        node.leaf = false;
        node.feature = best.feature;
        node.edges = best.edges;
        node.kids = new HashMap<>();

        // partition rows by bins
        Map<Integer, List<Integer>> parts = new HashMap<>();
        for (int r : rows) {
            int bin = digitize(X[r][best.feature], best.edges);
            parts.computeIfAbsent(bin, k -> new ArrayList<>()).add(r);
        }
        // build children
        for (var e : parts.entrySet()){
            int[] childIdx = e.getValue().stream().mapToInt(i->i).toArray();
            node.kids.put(e.getKey(), build(X, y, childIdx, depth+1, p));
        }
        return node;
    }

    private double predictOne(Node node, double[] x){
        Node cur = node;
        while (!cur.leaf){
            int bin = digitize(x[cur.feature], cur.edges);
            Node nxt = cur.kids.get(bin);
            if (nxt == null) { // unseen bin → backoff: majority at this node
                // walk to a child with largest population if available
                // (kids’ leaves will have majority labels)
                double fallback = 0.0; boolean set=false;
                for (Node c : cur.kids.values()){
                    if (c.leaf){ fallback = c.label; set=true; break; }
                }
                return set ? fallback : 0.0;
            }
            cur = nxt;
        }
        return cur.label;
    }

    // ---------- Splitting utilities ----------

    private static boolean pure(double[] y, int[] rows){
        double f = y[rows[0]];
        for (int r : rows) if (y[r] != f) return false;
        return true;
    }

    private static double majority(double[] y, int[] rows){
        Map<Double,Integer> cnt = new HashMap<>();
        for (int r: rows) cnt.merge(y[r], 1, Integer::sum);
        double best = 0; int bc = -1;
        for (var e: cnt.entrySet()){
            if (e.getValue() > bc) { bc = e.getValue(); best = e.getKey(); }
        }
        return best;
    }

    private static class Best { int feature; double[] edges; double gain; }

    private Best bestSplit(double[][] X, double[] y, int[] rows, int p, int nBins){
        double baseH = entropy(y, rows);
        Best best = null;

        for (int j=0;j<p;j++){
            double[] col = new double[rows.length];
            for (int t=0;t<rows.length;t++) col[t] = X[rows[t]][j];

            double[] edges = histogramEdges(col, nBins);
            // assign bins
            int[] bins = new int[rows.length];
            for (int t=0;t<rows.length;t++) bins[t] = digitize(col[t], edges);

            double g = informationGain(y, rows, bins);
            if (g > 0){
                double gain = baseH - (baseH - g); // (equivalently just g)
                if (best == null || gain > best.gain){
                    best = new Best(); best.feature = j; best.edges = edges; best.gain = gain;
                }
            }
        }
        return best;
    }

    private static double entropy(double[] y, int[] rows){
        Map<Double,Integer> cnt = new HashMap<>();
        for (int r: rows) cnt.merge(y[r], 1, Integer::sum);
        double n = rows.length, H = 0.0;
        for (int c: cnt.values()){
            double p = c / n;
            H += -p * log2(p);
        }
        return H;
    }

    private static double informationGain(double[] y, int[] rows, int[] bins){
        // IG = H(Y) - sum_v p(v) * H(Y|v)
        double H = entropy(y, rows);
        Map<Integer, List<Integer>> byV = new HashMap<>();
        for (int i=0;i<rows.length;i++){
            byV.computeIfAbsent(bins[i], k->new ArrayList<>()).add(rows[i]);
        }
        double cond = 0.0, n = rows.length;
        for (var e: byV.entrySet()){
            int[] subset = e.getValue().stream().mapToInt(k->k).toArray();
            cond += (subset.length / n) * entropy(y, subset);
        }
        return H - cond;
    }

    private static double[] histogramEdges(double[] col, int bins){
        double min = Double.POSITIVE_INFINITY, max = Double.NEGATIVE_INFINITY;
        for (double v: col){ if (v<min) min=v; if (v>max) max=v; }
        if (min==max){ // constant column; make a single split edge to produce one bin
            return new double[]{min};
        }
        double[] edges = new double[bins+1];
        double step = (max - min) / bins;
        for (int i=0;i<=bins;i++) edges[i] = min + i*step;
        return edges;
    }

    /** Return 1..bins index like numpy.digitize (right-closed). */
    private static int digitize(double v, double[] edges){
        // edges length = bins+1
        if (v <= edges[0]) return 1;
        if (v >= edges[edges.length-1]) return edges.length; // last bin index = bins+1
        int lo = 0, hi = edges.length-1;
        while (lo+1 < hi){
            int mid = (lo+hi)/2;
            if (v <= edges[mid]) hi = mid; else lo = mid;
        }
        return hi; // 1..bins
    }

    private static double log2(double x){ return Math.log(x + 1e-12) / Math.log(2.0); }
}
