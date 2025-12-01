package ml.core;

public interface Model {
    String name();
    void fit(Dataset train);
    double[] predict(double[][] X);       // class ids for classifiers; numeric for regressors
    default boolean isClassifier() { return true; }
}