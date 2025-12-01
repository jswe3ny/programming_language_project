package ml.models;

import ml.core.Dataset;
import ml.core.Model;

/** Linear Regression (closed-form / normal equation) with optional L2 on weights (bias unpenalized). */
public class LinearRegression implements Model {
    private double[] w;   // weights (n_features)
    private double b;     // bias
    private final double l2;

    public LinearRegression(double l2){ this.l2 = Math.max(0.0, l2); }

    @Override public String name(){ return "LinearRegression(l2="+l2+")"; }
    @Override public boolean isClassifier(){ return false; }

    @Override public void fit(Dataset d){
        int n = d.nRows(), p = d.nCols();

        // X_aug = [X | 1]
        double[][] Xa = new double[n][p+1];
        for (int i=0;i<n;i++){
            System.arraycopy(d.X[i], 0, Xa[i], 0, p);
            Xa[i][p] = 1.0; // bias column
        }

        double[][] XtX = new double[p+1][p+1];
        double[]   Xty = new double[p+1];

        // XtX = Xa^T * Xa
        for (int i=0;i<n;i++){
            double[] xi = Xa[i];
            for (int a=0;a<p+1;a++){
                Xty[a] += xi[a] * d.y[i];
                double va = xi[a];
                for (int b=0;b<p+1;b++){
                    XtX[a][b] += va * xi[b];
                }
            }
        }
        // Add L2 to weights only (not bias entry)
        for (int j=0;j<p;j++) XtX[j][j] += l2;

        // Solve (XtX) * beta = Xty  via Gaussian elimination (small system; OK here)
        double[] beta = solveSymmetric(XtX, Xty); // beta[0..p-1]=w, beta[p]=b
        w = new double[p];
        System.arraycopy(beta, 0, w, 0, p);
        b = beta[p];
    }

    @Override public double[] predict(double[][] X){
        double[] out = new double[X.length];
        for (int i=0;i<X.length;i++){
            double s = b;
            for (int j=0;j<w.length;j++) s += w[j]*X[i][j];
            out[i] = s;
        }
        return out;
    }

    // Simple solver for (A)x = b; A assumed SPD-ish (from XtX + l2I). Pivoting kept simple.
    private static double[] solveSymmetric(double[][] A, double[] b){
        int n = b.length;
        double[][] M = new double[n][n];
        double[]    y = new double[n];
        for (int i=0;i<n;i++){ System.arraycopy(A[i],0,M[i],0,n); y[i]=b[i]; }

        // Gaussian elimination with partial pivoting
        for (int k=0;k<n;k++){
            int piv=k; for (int i=k+1;i<n;i++) if (Math.abs(M[i][k])>Math.abs(M[piv][k])) piv=i;
            if (Math.abs(M[piv][k]) < 1e-12) continue;
            if (piv!=k){
                double[] tmp = M[k]; M[k]=M[piv]; M[piv]=tmp;
                double t = y[k]; y[k]=y[piv]; y[piv]=t;
            }
            double diag = M[k][k];
            for (int j=k;j<n;j++) M[k][j] /= diag;
            y[k] /= diag;
            for (int i=k+1;i<n;i++){
                double f = M[i][k];
                if (f==0) continue;
                for (int j=k;j<n;j++) M[i][j] -= f*M[k][j];
                y[i] -= f*y[k];
            }
        }
        // back substitution
        double[] x = new double[n];
        for (int i=n-1;i>=0;i--){
            double s = y[i];
            for (int j=i+1;j<n;j++) s -= M[i][j]*x[j];
            x[i]=s;
        }
        return x;
    }
}
