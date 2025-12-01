package ml.models;

import ml.core.Dataset;
import ml.core.Model;
import java.util.LinkedHashMap;

/** Gaussian Naive Bayes for binary/multi-class classification.
 *  Works on numeric features (your pipeline already OHEs categoricals).
 */
public class GaussianNaiveBayes implements Model {
    private double[] classes;                 // distinct class labels (e.g., {0.0, 1.0})
    private double[] priors;                  // p(c)
    private double[][] means;                 // means[c][j]
    private double[][] vars;                  // variances[c][j] (with smoothing)
    private final double varSmoothing;

    public GaussianNaiveBayes(double varSmoothing){
        this.varSmoothing = varSmoothing <= 0 ? 1e-9 : varSmoothing;
    }

    @Override public String name(){ return "GaussianNB(smooth="+varSmoothing+")"; }

    @Override public void fit(Dataset d){
    // collect unique class labels, stable order
    LinkedHashMap<Double, Integer> map = new LinkedHashMap<>();
    for (double v: d.y) map.putIfAbsent(v, map.size());
    int C = map.size(), P = d.nCols();
    classes = new double[C];
    priors  = new double[C];
    means   = new double[C][P];
    vars    = new double[C][P];

    // count per class
    int[] count = new int[C];
    for (double v: d.y) count[map.get(v)]++;
    int N = d.nRows();

    // priors + means (first pass)
    for (int i=0;i<N;i++){
        int c = map.get(d.y[i]);
        priors[c] += 1.0;
        double[] xi = d.X[i];
        for (int j=0;j<P;j++) means[c][j] += xi[j];
    }
    for (int c=0;c<C;c++){
        classes[c] = getKeyByIndex(map, c);
        if (count[c] == 0) count[c] = 1;               // guard
        priors[c] /= N;
        for (int j=0;j<P;j++) means[c][j] /= count[c];
    }

    // variances (second pass)
    for (int i=0;i<N;i++){
        int c = map.get(d.y[i]);
        double[] xi = d.X[i];
        for (int j=0;j<P;j++){
            double z = xi[j] - means[c][j];
            vars[c][j] += z*z;
        }
    }
    for (int c=0;c<C;c++){
        for (int j=0;j<P;j++){
            double v = vars[c][j] / Math.max(1, count[c]);
            vars[c][j] = v + varSmoothing;              // smoothing
        }
    }
    }

    @Override public double[] predict(double[][] X){
    int n = X.length, C = classes.length, P = means[0].length;
    double[] out = new double[n];
    for (int i=0;i<n;i++){
        double best = Double.NEGATIVE_INFINITY;
        int bestC = 0;
        for (int c=0;c<C;c++){
            // log p(c) + sum_j log N(x_j; mu, var)
            double lp = Math.log(priors[c] + 1e-12);
            for (int j=0;j<P;j++){
                double v = vars[c][j];
                double z = X[i][j] - means[c][j];
                lp += -0.5 * Math.log(2*Math.PI*v) - 0.5 * (z*z)/v;
            }
            if (lp > best){ best = lp; bestC = c; }
        }
        out[i] = classes[bestC];
    }
    return out;
    }

    @Override public boolean isClassifier(){ return true; }

    private static double getKeyByIndex(LinkedHashMap<Double,Integer> map, int idx){
        for (java.util.Map.Entry<Double, Integer> e : map.entrySet()) {
            if (e.getValue() == idx) return e.getKey();
        }
        return 0.0;
    }
}
