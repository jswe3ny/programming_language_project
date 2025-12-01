package ml.metrics;

import java.util.*;
public class Metrics {

    // ============================================================
    // CLASSIFICATION METRICS
    // ============================================================

    // Accuracy: percent of exact matches
    public static double accuracy(double[] y, double[] pred) {
        int n = y.length;
        int correct = 0;
        for (int i = 0; i < n; i++) {
            double p = pred[i] >= 0.5 ? 1.0 : 0.0;
            if (p == y[i]) correct++;
        }
        return correct / (double) n;
    }

    // Macro-F1 for binary/multiclass classification
    public static double macroF1(double[] y, double[] pred) {
        // Convert predictions to discrete {0,1}
        double[] p = new double[pred.length];
        for (int i = 0; i < p.length; i++)
            p[i] = pred[i] >= 0.5 ? 1.0 : 0.0;

        // Find classes
        Set<Double> labels = new LinkedHashSet<>();
        for (double v : y) labels.add(v);

        double sumF1 = 0.0;
        int count = 0;

        for (double c : labels) {
            int tp = 0, fp = 0, fn = 0;

            for (int i = 0; i < y.length; i++) {
                boolean truth = (y[i] == c);
                boolean predC = (p[i] == c);

                if (truth && predC) tp++;
                else if (!truth && predC) fp++;
                else if (truth && !predC) fn++;
            }

            double precision = tp + fp == 0 ? 0.0 : (tp / (double)(tp + fp));
            double recall    = tp + fn == 0 ? 0.0 : (tp / (double)(tp + fn));

            double f1 = (precision + recall == 0) ? 0.0 :
                (2 * precision * recall) / (precision + recall);

            sumF1 += f1;
            count++;
        }
        return sumF1 / count;
    }


    // ============================================================
    // REGRESSION METRICS
    // ============================================================

    // RMSE: sqrt(mean squared error)
    public static double rmse(double[] y, double[] pred) {
        int n = y.length;
        double s = 0;
        for (int i = 0; i < n; i++) {
            double d = y[i] - pred[i];
            s += d * d;
        }
        return Math.sqrt(s / n);
    }

    // R^2 coefficient of determination
    public static double r2(double[] y, double[] pred) {
        int n = y.length;

        // Mean of y
        double mean = 0;
        for (double v : y) mean += v;
        mean /= n;

        double ssTot = 0, ssRes = 0;
        for (int i = 0; i < n; i++) {
            ssTot += (y[i] - mean) * (y[i] - mean);
            ssRes += (y[i] - pred[i]) * (y[i] - pred[i]);
        }

        if (ssTot == 0) return 0.0;  // degenerate case
        return 1.0 - (ssRes / ssTot);
    }
}
