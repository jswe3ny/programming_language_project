package ml.models;

import ml.core.Dataset;
import ml.core.Model;
import java.util.Arrays;
import java.util.Random;

/** k-Nearest Neighbors (classification). */
public class KNN implements Model {
    private double[][] Xtr;
    private double[] ytr;

    private final int k;
    private final String distance;
    private final boolean weighted;
    private final String tieBreak;
    private final long seed;

    public KNN(int k, String distance, boolean weighted, String tieBreak, long seed) {
        if (k <= 0) throw new IllegalArgumentException("k must be >= 1");
        this.k = k;
        this.distance = (distance == null ? "euclidean" : distance.toLowerCase());
        this.weighted = weighted;
        this.tieBreak = (tieBreak == null ? "smallest" : tieBreak.toLowerCase());
        this.seed = seed;
    }

    @Override
    public String name() {
        return "kNN(k=" + k + ", " + distance + (weighted ? ", weighted" : "") + ")";
    }

    @Override
    public void fit(Dataset train) {
        this.Xtr = train.X;
        this.ytr = train.y;
    }

    @Override
    public double[] predict(double[][] X) {
        if (Xtr == null) throw new IllegalStateException("Call fit() first.");

        double[][] D;

        // Java-8 compatible replacement for switch-expression
        if (distance.equals("euclidean")) {
            D = euclidean(X, Xtr);
        } else if (distance.equals("manhattan")) {
            D = manhattan(X, Xtr);
        } else {
            throw new IllegalArgumentException("distance must be euclidean|manhattan");
        }

        int n = X.length;
        double[] out = new double[n];
        Random rng = new Random(seed);

        for (int i = 0; i < n; i++) {
            int[] idx = argsortRow(D[i], k);
            out[i] = vote(idx, D[i], rng);
        }
        return out;
    }

    private double vote(int[] neighIdx, double[] dRow, Random rng) {
        double[] labs = new double[neighIdx.length];
        for (int i = 0; i < neighIdx.length; i++) labs[i] = ytr[neighIdx[i]];

        Arrays.sort(labs);
        int m = 0;
        for (int i = 0; i < labs.length; i++)
            if (i == 0 || labs[i] != labs[i - 1]) labs[m++] = labs[i];
        double[] uniq = Arrays.copyOf(labs, m);

        double[] scores = new double[m];

        if (weighted) {
            double eps = 1e-12;
            for (int j = 0; j < neighIdx.length; j++) {
                double lab = ytr[neighIdx[j]];
                double w = 1.0 / (dRow[neighIdx[j]] + eps);
                int pos = indexOf(uniq, lab);
                scores[pos] += w;
            }
        } else {
            for (int j = 0; j < neighIdx.length; j++) {
                double lab = ytr[neighIdx[j]];
                int pos = indexOf(uniq, lab);
                scores[pos] += 1.0;
            }
        }

        double best = Double.NEGATIVE_INFINITY;
        int bestIdx = -1;

        for (int i = 0; i < scores.length; i++)
            if (scores[i] > best) { best = scores[i]; bestIdx = i; }

        int ties = 0;
        for (double s : scores) if (s == best) ties++;

        if (ties > 1) {
            if (tieBreak.equals("random")) {
                int[] tiePos = new int[ties];
                int t = 0;
                for (int i = 0; i < scores.length; i++)
                    if (scores[i] == best) tiePos[t++] = i;
                bestIdx = tiePos[rng.nextInt(ties)];
            } else {
                double minLab = Double.POSITIVE_INFINITY;
                int minIdx = -1;
                for (int i = 0; i < scores.length; i++)
                    if (scores[i] == best && uniq[i] < minLab) {
                        minLab = uniq[i];
                        minIdx = i;
                    }
                bestIdx = minIdx;
            }
        }

        return uniq[bestIdx];
    }

    private static int indexOf(double[] arr, double val) {
        for (int i = 0; i < arr.length; i++) if (arr[i] == val) return i;
        return -1;
    }

    private static int[] argsortRow(double[] row, int k) {
        int n = row.length;
        int[] idx = new int[n];
        for (int i = 0; i < n; i++) idx[i] = i;

        for (int i = 0; i < k; i++) {
            int minPos = i;
            for (int j = i + 1; j < n; j++)
                if (row[idx[j]] < row[idx[minPos]]) minPos = j;
            int tmp = idx[i]; idx[i] = idx[minPos]; idx[minPos] = tmp;
        }

        return Arrays.copyOf(idx, k);
    }

    private static double[][] euclidean(double[][] A, double[][] B) {
        int nt = A.length, nb = B.length, d = A[0].length;
        double[][] D = new double[nt][nb];
        for (int i = 0; i < nt; i++)
            for (int j = 0; j < nb; j++) {
                double s = 0;
                for (int k = 0; k < d; k++) {
                    double z = A[i][k] - B[j][k];
                    s += z * z;
                }
                D[i][j] = Math.sqrt(Math.max(s, 0.0));
            }
        return D;
    }

    private static double[][] manhattan(double[][] A, double[][] B) {
        int nt = A.length, nb = B.length, d = A[0].length;
        double[][] D = new double[nt][nb];
        for (int i = 0; i < nt; i++)
            for (int j = 0; j < nb; j++) {
                double s = 0;
                for (int k = 0; k < d; k++)
                    s += Math.abs(A[i][k] - B[j][k]);
                D[i][j] = s;
            }
        return D;
    }

    @Override
    public boolean isClassifier() {
        return true;
    }
}
