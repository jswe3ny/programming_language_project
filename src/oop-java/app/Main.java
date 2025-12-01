
package app;

import java.util.Scanner;
import ml.core.*; 
import ml.metrics.*; 
import ml.models.*; 
import ml.preprocess.*; 
import ml.helpers.*;

public class Main {
    static Dataset TRAIN, TEST;

    public static void main(String[] args) throws Exception {
        ArgParser ap = new ArgParser(args);
        try (Scanner in = new Scanner(System.in)) {

        System.out.println("\nMenu:\n" +
        "(1) Load data\n" +
        "(2) Linear Regression (closed-form)\n" +
        "(3) Logistic Regression (binary)\n" +
        "(4) k-Nearest Neighbors\n" +
        "(5) Decision Tree \n" +
        "(6) Gaussian Naive Bayes6\n" + 
        "(7) Print general results\n" +
        "(8) Quit");


            while(true){
                System.out.print("Enter option: ");
                int opt = Integer.parseInt(in.nextLine().trim());
                if(opt==8) break;
                if(opt==1) loadData(ap);
                if(opt==2) runLinear(ap);
                if(opt==3) runLogistic(ap);
                if(opt==4) runKNN(ap);
                if(opt==5) runTree(ap);
                if(opt==6) runGNB(ap);
                if(opt==7) printResults();
            }
        }
    }

    private static void loadData(ArgParser ap) throws Exception {
        String path = ap.get("train", "adult_income_cleaned.csv"); // adjust if needed
        boolean normalize = ap.has("normalize");                      // pass --normalize to enable
        System.out.println("Loading "+path+" (normalize="+normalize+") ...");
        long t0 = System.nanoTime();
        var tt = AdultPipeline.loadClassification(path, "income", normalize, 0.2, 42);
        TRAIN = tt.train; TEST = tt.test;
        double secs = (System.nanoTime()-t0)/1e9;
        System.out.printf("Loaded: %d train / %d test / %d features in %.3fs%n",
                TRAIN.nRows(), TEST.nRows(), TRAIN.nCols(), secs);
    }


    private static void runLinear(ArgParser ap){
        // Use regression target; default per spec is hours.per.week (sometimes header uses dots)
        String target = ap.get("target", "hours.per.week");
        boolean normalize = ap.has("normalize"); // scaling helps a bit here too
        String path = ap.get("train", "adult_income_cleaned.csv");
        String srcRoot = ap.get("srcroot", "src");


        try {
            var tt = AdultPipeline.loadRegression(path, target, normalize, 0.2, 42);
            Timer t = new Timer(); t.start();
            // L2=0 by default; pass --l2 0.001 to try ridge
            double l2 = ap.getDouble("l2", 0.0);
            ml.models.LinearRegression m = new ml.models.LinearRegression(l2);
            m.fit(tt.train);
            double secs = t.seconds();
            double[] pred = m.predict(tt.test.X);
            double acc = Metrics.accuracy(TEST.y, pred);
            double f1  = Metrics.macroF1(TEST.y, pred);
            int sloc = SLOC.forClass(LogisticRegression.class, srcRoot);


            System.out.println("Algorithm: " + m.name());
            System.out.printf("Train time: %.4f s%n", secs);
            System.out.printf("RMSE: %.4f%n", ml.metrics.Metrics.rmse(tt.test.y, pred));
            System.out.printf("R^2: %.4f%n", ml.metrics.Metrics.r2(tt.test.y, pred));
            System.out.println("SLOC: " + sloc);
            log(m.name(), secs, "Accuracy", acc, "Macro-F1", f1, sloc);
        } catch (Exception e){
            System.out.println("Linear regression run failed: " + e.getMessage());
            e.printStackTrace();
        }
    }

    private static void runLogistic(ArgParser ap){
        double lr = ap.getDouble("lr", 0.2);
        int epochs = ap.getInt("epochs", 400);
        double l2 = ap.getDouble("l2", 1e-3);
        long seed = 7;
        String srcRoot = ap.get("srcroot", "src");


        if (TRAIN==null){ 
            System.out.println("Load data first (option 1)."); 
            return; 
        }
        Timer t = new Timer(); t.start();
        LogisticRegression m = new LogisticRegression(lr, epochs, l2, seed);
        m.fit(TRAIN);
        double secs = t.seconds();
        double[] pred = m.predict(TEST.X);
        double acc = Metrics.accuracy(TEST.y, pred);
        double f1  = Metrics.macroF1(TEST.y, pred);
        int sloc = SLOC.forClass(LogisticRegression.class, srcRoot);

        System.out.println("Algorithm: "+m.name());
        System.out.printf("Train time: %.4f s%n", secs);
        System.out.printf("Accuracy: %.4f%n", Metrics.accuracy(TEST.y, pred));
        System.out.printf("Macro-F1: %.4f%n", Metrics.macroF1(TEST.y, pred));
        System.out.println("SLOC: " + sloc);
        log(m.name(), secs, "Accuracy", acc, "Macro-F1", f1, sloc);
    }

    private static void runKNN(ml.helpers.ArgParser ap){
        if (TRAIN==null){ 
            System.out.println("Load data first (option 1)."); 
            return; 
        }
        int k = ap.getInt("k", 7);
        String distance = ap.get("distance", "euclidean");  // or "manhattan"
        boolean weighted = ap.has("weighted");               // pass --weighted to enable
        String tie = ap.get("tie", "smallest");              // or "random"
        int sloc = SLOC.forClass(KNN.class, ap.get("srcroot","src"));


        ml.helpers.Timer t = new ml.helpers.Timer(); t.start();
        ml.models.KNN m = new ml.models.KNN(k, distance, weighted, tie, 0L);
        m.fit(TRAIN);
        double secs = t.seconds();
        double[] pred = m.predict(TEST.X);
        double acc = Metrics.accuracy(TEST.y, pred);
        double f1  = Metrics.macroF1(TEST.y, pred);

        System.out.println("Algorithm: " + m.name());
        System.out.printf("Train time: %.4f s%n", secs); // kNN "train" is just storing data
        System.out.printf("Accuracy: %.4f%n", ml.metrics.Metrics.accuracy(TEST.y, pred));
        System.out.printf("Macro-F1: %.4f%n", ml.metrics.Metrics.macroF1(TEST.y, pred));
        System.out.println("SLOC: " + sloc);
        log(m.name(), secs, "Accuracy", acc, "Macro-F1", f1, sloc);
    }

    private static void runTree(ArgParser ap){
        if (TRAIN==null){ System.out.println("Load data first (option 1)."); return; }
        int maxDepth   = ap.getInt("max_depth", 5);
        int minSamples = ap.getInt("min_samples", 10);
        int nBins      = ap.getInt("bins", 16);

        Timer t = new Timer(); t.start();
        ml.models.DecisionTree m = new ml.models.DecisionTree(maxDepth, minSamples, nBins);
        m.fit(TRAIN);
        double secs = t.seconds();
        double[] pred = m.predict(TEST.X);
        double acc = Metrics.accuracy(TEST.y, pred);
        double f1  = Metrics.macroF1(TEST.y, pred);
        int sloc = SLOC.forClass(DecisionTree.class, ap.get("srcroot","src"));


        System.out.println("Algorithm: " + m.name());
        System.out.printf("Train time: %.4f s%n", secs);
        System.out.printf("Accuracy: %.4f%n", Metrics.accuracy(TEST.y, pred));
        System.out.printf("Macro-F1: %.4f%n", Metrics.macroF1(TEST.y, pred));
        System.out.println("SLOC: " + sloc);
        log(m.name(), secs, "Accuracy", acc, "Macro-F1", f1, sloc);
    }


    private static void runGNB(ArgParser ap){
        if (TRAIN==null){ System.out.println("Load data first (option 1)."); return; }
        double smooth = ap.getDouble("smoothing", 1e-9);   // --smoothing 1e-8, etc.

        Timer t = new Timer(); t.start();
        ml.models.GaussianNaiveBayes m = new ml.models.GaussianNaiveBayes(smooth);
        m.fit(TRAIN);
        double secs = t.seconds();
        double[] pred = m.predict(TEST.X);
        double acc = Metrics.accuracy(TEST.y, pred);
        double f1  = Metrics.macroF1(TEST.y, pred);
        int sloc = SLOC.forClass(GaussianNaiveBayes.class, ap.get("srcroot","src"));

        System.out.println("Algorithm: " + m.name());
        System.out.printf("Train time: %.4f s%n", secs);
        System.out.printf("Accuracy: %.4f%n", Metrics.accuracy(TEST.y, pred));
        System.out.printf("Macro-F1: %.4f%n", Metrics.macroF1(TEST.y, pred));
        System.out.println("SLOC: " + sloc);
        log(m.name(), secs, "Accuracy", acc, "Macro-F1", f1, sloc);

    }

    static class Row {
        String impl, algo; double t, m1, m2; String m1n, m2n; int sloc;
        Row(String impl, String algo, double t, String m1n, double m1, String m2n, double m2, int sloc){
            this.impl=impl; this.algo=algo; this.t=t; this.m1n=m1n; this.m1=m1; this.m2n=m2n; this.m2=m2; this.sloc=sloc;
        }
    }
    static java.util.List<Row> RESULTS = new java.util.ArrayList<>();
    static void log(String algo, double secs, String m1n, double m1, String m2n, double m2, int sloc){
        RESULTS.add(new Row("Java", algo, secs, m1n, m1, m2n, m2, sloc));
    }
    static void printResults(){
        System.out.println("General Results (Comparison):");
        System.out.println("Impl     Algo                   TrainTime   Metric1              Metric2              SLOC");
        for (Row r: RESULTS){
            System.out.printf("%-8s %-22s %8.4fs   %-10s %-8.4f   %-10s %-8.4f   %5d%n",
            r.impl, r.algo, r.t, r.m1n, r.m1, r.m2n, r.m2, r.sloc);
        }
    }
}
