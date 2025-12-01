package ml.helpers;

public class Timer {
    private long t0;
    public void start(){ 
        t0 = System.nanoTime(); 
    }
    public double seconds(){ 
        return (System.nanoTime()-t0)/1_000_000_000.0; 
    }
}