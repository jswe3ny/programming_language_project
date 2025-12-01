package ml.models;
import ml.core.*;

public class LogisticRegression implements Model {
    private double[] w; private double b;
    private final double lr; private final int epochs; private final double l2; private final long seed;
    public LogisticRegression(double lr, int epochs, double l2, long seed){ this.lr=lr; this.epochs=epochs; this.l2=l2; this.seed=seed; }
    public String name(){ return "LogisticRegression(GD)"; }

    public void fit(Dataset d){
        int n=d.nRows(), p=d.nCols(); w=new double[p];
        java.util.Random r=new java.util.Random(seed);
        for(int j=0;j<p;j++) w[j]=(r.nextDouble()-0.5)*1e-2; b=0;
        for(int e=0;e<epochs;e++){
            double[] gw=new double[p]; double gb=0;
            for(int i=0;i<n;i++){
                double z=b; for(int j=0;j<p;j++) z+=w[j]*d.X[i][j];
                double y=d.y[i]; double p1=1.0/(1.0+Math.exp(-z));
                double g = (p1 - y);
                for(int j=0;j<p;j++) gw[j]+= g*d.X[i][j];
                gb += g;
            }
            for(int j=0;j<p;j++) gw[j]=gw[j]/n + l2*w[j];
            gb/=n;
            for(int j=0;j<p;j++) w[j]-=lr*gw[j]; b -= lr*gb;
        }
    }
    public double[] predict(double[][] X){
        double[] out=new double[X.length];
        for(int i=0;i<X.length;i++){
            double z=b; for(int j=0;j<w.length;j++) z+=w[j]*X[i][j];
            out[i] = (1.0/(1.0+Math.exp(-z)) >= 0.5) ? 1.0 : 0.0;
        }
        return out;
    }
}