package ml.core;
import java.util.Arrays;
import java.util.Random;

public class Dataset {
    public final double[][] X;
    public final double[] y;
    public final String[] featureNames;
    public final String targetName;

    public Dataset(double[][] X, double[] y, String[] featureNames, String targetName) {
        this.X = X; this.y = y; this.featureNames = featureNames; this.targetName = targetName;
    }

    public int nRows(){ return X.length; }
    public int nCols(){ return X.length==0?0:X[0].length; }

    public static class TrainTest { 
        public final Dataset train, test; 
        public TrainTest(Dataset a, Dataset b){train=a;test=b;} 
    }

    public static TrainTest split(Dataset full, double testFrac, long seed){
        int n = full.nRows(); int t = (int)Math.round(n*testFrac);
        int[] idx = new int[n]; for(int i=0;i<n;i++) idx[i]=i;
        Random r = new Random(seed);
        for(int i=n-1;i>0;i--){ int j=r.nextInt(i+1); int tmp=idx[i]; idx[i]=idx[j]; idx[j]=tmp; }
        int[] testIdx = Arrays.copyOfRange(idx,0,t), trainIdx = Arrays.copyOfRange(idx,t,n);
        return new TrainTest(select(full, trainIdx), select(full, testIdx));
    }

    private static Dataset select(Dataset d, int[] idx){
        double[][] X = new double[idx.length][d.nCols()];
        double[] y = new double[idx.length];
        for(int i=0;i<idx.length;i++){ X[i]=Arrays.copyOf(d.X[idx[i]], d.nCols()); y[i]=d.y[idx[i]]; }
        return new Dataset(X,y,d.featureNames,d.targetName);
    }
}