package ml.metrics;

import java.util.*;

public class Metrics {
    public static double accuracy(double[] y, double[] p){
        int n=y.length, c=0; 
        for(int i=0;i<n;i++) 
            if((int)y[i]==(int)p[i]) c++; 
            return c/(double)n;
    }
    public static double macroF1(double[] y, double[] p){
        Set<Integer> labs=new HashSet<>(); 
        for(double v:y) 
        labs.add((int)v); 
        for(double v:p) 
        labs.add((int)v);
        double s=0; 
        for(int L: labs) s += f1ForLabel(y,p,L); 
        
        return s/labs.size();
    }
    private static double f1ForLabel(double[] y, double[] p, int L){
        double tp=0, fp=0, fn=0;
        for(int i=0;i<y.length;i++){
            boolean yy=((int)y[i])==L, pp=((int)p[i])==L;
            if(pp&&yy) tp++; 
            else if(pp) fp++; 
            else if(yy) fn++;
        }
        double prec = tp==0?0:tp/(tp+fp);
        double rec  = tp==0?0:tp/(tp+fn);
        return (prec+rec)==0?0:2*prec*rec/(prec+rec);
    }
    public static double rmse(double[] y, double[] p){
        double se=0; 
        for(int i=0;i<y.length;i++){ 
            double d=p[i]-y[i]; se+=d*d; 
        } 
        return Math.sqrt(se/y.length);
    }
    public static double r2(double[] y, double[] p){
        double mean=0; 
        for(double v:y) 
        mean+=v; 
        mean/=y.length;
        
        double sst=0, ssr=0; 
        for(int i=0;i<y.length;i++){ 
            double d=y[i]-mean; 
            sst+=d*d; 
            double e=y[i]-p[i]; 
            ssr+=e*e; 
        }
        return 1.0 - ssr/(sst+1e-12);
    }
}