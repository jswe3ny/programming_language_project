package ml.models;
import ml.core.*;


public class LogisticRegression implements Model {

    private double[] w;  // weights
    private double b;    // bias

    private final double lr;      // learning rate
    private final int epochs;     // number of passes
    private final double l2;      // L2 regularization strength
    private final long seed;      // RNG seed for initialization

    public LogisticRegression(double lr, int epochs, double l2, long seed) {
        this.lr = lr;
        this.epochs = epochs;
        this.l2 = l2;
        this.seed = seed;
    }

    @Override
    public String name() {
        return "LogisticRegression(GD)";
    }

    @Override
    public void fit(Dataset d) {
        int n = d.nRows();
        int p = d.nCols();

        w = new double[p];
        java.util.Random r = new java.util.Random(seed);

        // small random initialization
        for (int j = 0; j < p; j++) {
            w[j] = (r.nextDouble() - 0.5) * 1e-2;
        }
        b = 0.0;

        // gradient descent loop
        for (int e = 0; e < epochs; e++) {

            double[] gw = new double[p];
            double gb = 0.0;

            for (int i = 0; i < n; i++) {

                // compute z = w*x + b
                double z = b;
                double[] xi = d.X[i];
                for (int j = 0; j < p; j++) {
                    z += w[j] * xi[j];
                }

                double y = d.y[i];
                double p1 = 1.0 / (1.0 + Math.exp(-z));  // sigmoid

                double g = (p1 - y); // gradient of logistic loss

                for (int j = 0; j < p; j++) {
                    gw[j] += g * xi[j];
                }
                gb += g;
            }

            // average + L2
            for (int j = 0; j < p; j++) {
                gw[j] = gw[j] / n + l2 * w[j];
            }
            gb /= n;

            // update parameters
            for (int j = 0; j < p; j++) {
                w[j] -= lr * gw[j];
            }
            b -= lr * gb;
        }
    }

    @Override
    public double[] predict(double[][] X) {
        double[] out = new double[X.length];

        for (int i = 0; i < X.length; i++) {
            double z = b;
            for (int j = 0; j < w.length; j++) {
                z += w[j] * X[i][j];
            }
            double p = 1.0 / (1.0 + Math.exp(-z));
            out[i] = (p >= 0.5) ? 1.0 : 0.0;
        }

        return out;
    }

    @Override
    public boolean isClassifier() {
        return true;
    }
}
