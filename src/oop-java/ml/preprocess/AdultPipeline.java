package ml.preprocess;
import ml.core.Dataset;
import java.io.*;
import java.util.*;

/** Reads cleaned Adult CSV, fits OHE+zscore on TRAIN, applies to TEST, maps income->{0,1}. */
public class AdultPipeline {

    public static Dataset.TrainTest loadClassification(String csvPath, String targetCol, boolean normalize, double testFrac, long seed) 
    
    throws IOException {
        Table t = readCSV(csvPath);
        int tIdx = indexOf(t.header, targetCol);
        if (tIdx < 0) throw new IllegalArgumentException("Target "+targetCol+" not found.");

        // split
        int n=t.rows.size(), testN=(int)Math.round(n*testFrac);
        int[] idx=perm(n,seed);
        List<String[]> testRows=new ArrayList<>(), trainRows=new ArrayList<>();
        for(int i=0;i<n;i++) ((i<testN)?testRows:trainRows).add(t.rows.get(idx[i]));

        // detect numeric on TRAIN
        Set<String> numeric = detectNumeric(t.header, trainRows, tIdx);

        // fit OHE cats on TRAIN
        LinkedHashMap<String,LinkedHashMap<String,Integer>> cats = fitCats(trainRows, t.header, numeric, tIdx);

        // final feature order: numeric first, then OHE blocks
        List<String> order = featureOrder(t.header, numeric, cats, tIdx);

        // build TRAIN
        double[][] Xtr = new double[trainRows.size()][order.size()];
        double[] ytr   = new double[trainRows.size()];
        for(int i=0;i<trainRows.size();i++){
            RowEnc r = encodeRow(trainRows.get(i), t.header, order, numeric, cats, tIdx);
            Xtr[i]=r.x; ytr[i]=mapIncome(r.y);
        }

        // z-score fit on train
        double[] mean=new double[order.size()], std=new double[order.size()];
        Arrays.fill(std,1.0);
        if (normalize){ zfit(Xtr,mean,std); zapply(Xtr,mean,std); }

        // build TEST using TRAIN fit
        double[][] Xte = new double[testRows.size()][order.size()];
        double[] yte   = new double[testRows.size()];
        for(int i=0;i<testRows.size();i++){
            RowEnc r = encodeRow(testRows.get(i), t.header, order, numeric, cats, tIdx);
            if (normalize) zapply(r.x,mean,std);
            Xte[i]=r.x; yte[i]=mapIncome(r.y);
        }

        String[] featNames = order.toArray(new String[0]);
        Dataset train = new Dataset(Xtr,ytr,featNames,targetCol);
        Dataset test  = new Dataset(Xte,yte,featNames,targetCol);
        return new Dataset.TrainTest(train,test);
    }

    // --------- internals ----------
    private static class Table { String[] header; List<String[]> rows; }
    private static class RowEnc { double[] x; String y; RowEnc(double[] x, String y){this.x=x;this.y=y;} }

    private static Table readCSV(String path) throws IOException {
        try(BufferedReader br=new BufferedReader(new FileReader(path))){
            String head=br.readLine(); if(head==null) throw new IOException("Empty CSV");
            String[] header=head.split(",",-1);
            List<String[]> rows=new ArrayList<>();
            for(String line; (line=br.readLine())!=null; ) rows.add(line.split(",",-1));
            Table t=new Table(); t.header=header; t.rows=rows; return t;
        }
    }
    private static int indexOf(String[] a, String k){ for(int i=0;i<a.length;i++) if(a[i].equals(k)) return i; return -1; }
    private static int[] perm(int n,long s){ int[] i=new int[n]; for(int k=0;k<n;k++) i[k]=k; Random r=new Random(s);
        for(int k=n-1;k>0;k--){ int j=r.nextInt(k+1); int t=i[k]; i[k]=i[j]; i[j]=t; } return i; }
    private static boolean isNum(String s){ if(s==null||s.isEmpty()) return false; try{Double.parseDouble(s);}catch(Exception e){return false;} return true; }

    private static Set<String> detectNumeric(String[] header, List<String[]> rows, int tIdx){
        Set<String> num=new HashSet<>();
        for(int j=0;j<header.length;j++){
            if(j==tIdx) continue;
            boolean all=true;
            for(int i=0;i<Math.min(200, rows.size()); i++){
                if(!isNum(rows.get(i)[j].trim())) { all=false; break; }
            }
            if(all) num.add(header[j]);
        }
        return num;
    }

    private static LinkedHashMap<String,LinkedHashMap<String,Integer>> fitCats(List<String[]> rows, String[] header,
                                                                              Set<String> numeric, int tIdx){
        LinkedHashMap<String,LinkedHashMap<String,Integer>> cats=new LinkedHashMap<>();
        for(int j=0;j<header.length;j++){
            if(j==tIdx) continue;
            String col = header[j];
            if(numeric.contains(col)) continue;
            LinkedHashMap<String,Integer> map=new LinkedHashMap<>();
            for(String[] r: rows){ String v=r[j].trim(); map.computeIfAbsent(v, k->map.size()); }
            cats.put(col,map);
        }
        return cats;
    }

    private static List<String> featureOrder(String[] header, Set<String> numeric,
                                             LinkedHashMap<String,LinkedHashMap<String,Integer>> cats, int tIdx){
        List<String> order=new ArrayList<>();
        for(String h: header) if(!h.equals(header[tIdx]) && numeric.contains(h)) order.add(h);
        for(String col: cats.keySet()){
            for(String val: cats.get(col).keySet()) order.add(col+"=="+val);
        }
        return order;
    }

    private static RowEnc encodeRow(String[] row, String[] header, List<String> order,
                                    Set<String> numeric,
                                    LinkedHashMap<String,LinkedHashMap<String,Integer>> cats, int tIdx){
        double[] x=new double[order.size()];
        int pos=0;
        // numeric
        for(String h: header){
            if(h.equals(header[tIdx])) continue;
            if(numeric.contains(h)){
                double v=0; String s=row[indexOf(header,h)].trim();
                if(!s.isEmpty()) try{ v=Double.parseDouble(s);}catch(Exception ignored){}
                x[pos++]=v;
            }
        }
        // one-hot
        for(String col: cats.keySet()){
            LinkedHashMap<String,Integer> map = cats.get(col);
            int base=pos; pos += map.size();
            String v = row[indexOf(header,col)].trim();
            Integer idx = map.get(v); if(idx!=null) x[base+idx]=1.0; // unseen cats -> all zeros
        }
        return new RowEnc(x, row[tIdx]);
    }

    private static void zfit(double[][] X, double[] mean, double[] std){
        int n=X.length,d=X[0].length;
        for(int j=0;j>d?false:true;j++){
            double m=0; for(int i=0;i<n;i++) m+=X[i][j]; m/=n; mean[j]=m;
            double s=0; for(int i=0;i<n;i++){ double z=X[i][j]-m; s+=z*z; }
            std[j]=Math.sqrt(s/n)+1e-8;
        }
    }
    private static void zapply(double[][] X,double[] mean,double[] std){ for(double[] r:X) zapply(r,mean,std); }
    private static void zapply(double[] x,double[] mean,double[] std){ for(int j=0;j<x.length;j++) x[j]=(x[j]-mean[j])/std[j]; }

    private static double mapIncome(String raw){
        String s = raw==null? "" : raw.trim().replace(".","").toLowerCase();
        if ("<=50k".equals(s)) return 0.0; if (">50k".equals(s)) return 1.0; return 0.0;
    }

    public static Dataset.TrainTest loadRegression(String csvPath, String targetCol,
                                               boolean normalize, double testFrac, long seed) throws IOException {
    Table t = readCSV(csvPath);
    int tIdx = indexOf(t.header, targetCol);
    if (tIdx < 0) throw new IllegalArgumentException("Target "+targetCol+" not found.");

    // split
    int n=t.rows.size(), testN=(int)Math.round(n*testFrac);
    int[] idx=perm(n,seed);
    java.util.List<String[]> testRows=new java.util.ArrayList<>(), trainRows=new java.util.ArrayList<>();
    for(int i=0;i<n;i++) ((i<testN)?testRows:trainRows).add(t.rows.get(idx[i]));

    // detect numeric on TRAIN (for features, not target)
    java.util.Set<String> numeric = detectNumeric(t.header, trainRows, tIdx);

    // fit OHE on TRAIN
    java.util.LinkedHashMap<String,java.util.LinkedHashMap<String,Integer>> cats =
        fitCats(trainRows, t.header, numeric, tIdx);

    // final feature order
    java.util.List<String> order = featureOrder(t.header, numeric, cats, tIdx);

    // TRAIN
    double[][] Xtr = new double[trainRows.size()][order.size()];
    double[] ytr   = new double[trainRows.size()];
    for(int i=0;i<trainRows.size();i++){
        RowEnc r = encodeRow(trainRows.get(i), t.header, order, numeric, cats, tIdx);
        Xtr[i]=r.x; ytr[i]=parseDoubleSafe(trainRows.get(i)[tIdx]);
    }

    double[] mean=new double[order.size()], std=new double[order.size()];
    java.util.Arrays.fill(std,1.0);
    if (normalize){ zfit(Xtr,mean,std); zapply(Xtr,mean,std); }

    // TEST
    double[][] Xte = new double[testRows.size()][order.size()];
    double[] yte   = new double[testRows.size()];
    for(int i=0;i<testRows.size();i++){
        RowEnc r = encodeRow(testRows.get(i), t.header, order, numeric, cats, tIdx);
        if (normalize) zapply(r.x,mean,std);
        Xte[i]=r.x; yte[i]=parseDoubleSafe(testRows.get(i)[tIdx]);
    }

    String[] featNames = order.toArray(new String[0]);
    ml.core.Dataset train = new ml.core.Dataset(Xtr,ytr,featNames,targetCol);
    ml.core.Dataset test  = new ml.core.Dataset(Xte,yte,featNames,targetCol);
    return new ml.core.Dataset.TrainTest(train,test);
    }

    // small helper near other privates
    private static double parseDoubleSafe(String s){
        if (s==null || s.trim().isEmpty()) return 0.0;
        try { return Double.parseDouble(s.trim()); } catch(Exception e){ return 0.0; }
    }
}