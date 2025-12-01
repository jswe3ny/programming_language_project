package ml.preprocess;
import ml.core.Dataset;
import java.io.*;
import java.util.*;

public class AdultPipeline {

    // ============================================================
    // MAIN ENTRYPOINTS
    // ============================================================

    public static Dataset.TrainTest loadClassification(
            String csvPath,
            String targetCol,
            boolean normalize,
            double testFrac,
            long seed) throws IOException {

        Table t = readCSV(csvPath);

        // Map income -> 0/1
        double[] y = new double[t.nRows()];
        int ycol = t.colIndex(targetCol);
        for (int i = 0; i < y.length; i++) {
            String v = t.rows.get(i)[ycol].trim();
            if (v.equals(">50K")) y[i] = 1.0;
            else y[i] = 0.0;
        }

        // Remove target column from table for feature processing
        Table withoutY = t.removeColumn(targetCol);

        return makeTrainTest(withoutY, y, normalize, testFrac, seed);
            }


    public static Dataset.TrainTest loadRegression(
            String csvPath,
            String targetCol,
            boolean normalize,
            double testFrac,
            long seed) throws IOException {

        Table t = readCSV(csvPath);

        int ycol = t.colIndex(targetCol);
        double[] y = new double[t.nRows()];
        for (int i = 0; i < y.length; i++)
            y[i] = Double.parseDouble(t.rows.get(i)[ycol]);

        Table withoutY = t.removeColumn(targetCol);

        return makeTrainTest(withoutY, y, normalize, testFrac, seed);
            }

    // ============================================================
    // CORE PIPELINE
    // ============================================================

    private static Dataset.TrainTest makeTrainTest(
            Table tRaw,
            double[] y,
            boolean normalize,
            double testFrac,
            long seed) {

        // 1. Infer column types (numeric vs categorical)
        ColumnType[] types = inferTypes(tRaw);

        // 2. Split rows into train/test indices
        int N = tRaw.nRows();
        int testSize = (int) Math.round(N * testFrac);
        int trainSize = N - testSize;

        int[] idx = shuffledIndices(N, seed);
        int[] trainIdx = Arrays.copyOfRange(idx, 0, trainSize);
        int[] testIdx = Arrays.copyOfRange(idx, trainSize, N);

        // 3. Build train table
        Table trainT = tRaw.subset(trainIdx);
        double[] yTrain = subsetArray(y, trainIdx);

        // 4. Build test table
        Table testT = tRaw.subset(testIdx);
        double[] yTest = subsetArray(y, testIdx);

        // 5. Fit encoders using *train only*
        OneHotEncoder ohe = new OneHotEncoder(trainT, types);

        // 6. Encode train
        Encoded encTrain = ohe.encode(trainT, types);

        // 7. Encode test (align categories to train)
        Encoded encTest = ohe.encode(testT, types);

        // 8. Normalize numeric columns (optional)
        if (normalize) {
            Normalizer norm = new Normalizer(encTrain.X);
            norm.applyInPlace(encTrain.X);
            norm.applyInPlace(encTest.X);  // apply train stats
        }

        Dataset trainDs = new Dataset(encTrain.X, yTrain, encTrain.featureNames, "");
        Dataset testDs  = new Dataset(encTest.X,  yTest,  encTest.featureNames, "");

        return new Dataset.TrainTest(trainDs, testDs);
            }

    // ============================================================
    // COLUMN TYPE INFERRING
    // ============================================================

    private enum ColumnType { NUMERIC, CATEGORICAL }

    private static ColumnType[] inferTypes(Table t) {
        int C = t.nCols();
        ColumnType[] types = new ColumnType[C];

        for (int c = 0; c < C; c++) {
            // Try numeric parse on first 20 rows
            boolean numeric = true;
            for (int r = 0; r < Math.min(20, t.nRows()); r++) {
                String v = t.rows.get(r)[c].trim();
                if (!isNumeric(v)) { numeric = false; break; }
            }
            types[c] = numeric ? ColumnType.NUMERIC : ColumnType.CATEGORICAL;
        }
        return types;
    }

    private static boolean isNumeric(String s) {
        try { Double.parseDouble(s); return true; }
        catch (Exception e) { return false; }
    }

    // ============================================================
    // ENCODING SUPPORT
    // ============================================================

    private static class OneHotEncoder {
        Map<Integer, List<String>> categories = new HashMap<>();
        List<Integer> numericCols = new ArrayList<>();

        OneHotEncoder(Table trainT, ColumnType[] types) {
            int C = trainT.nCols();

            for (int c = 0; c < C; c++) {
                if (types[c] == ColumnType.NUMERIC) {
                    numericCols.add(c);
                } else {
                    // Collect distinct categories from training
                    Set<String> set = new LinkedHashSet<>();
                    for (String[] row : trainT.rows) {
                        set.add(row[c].trim());
                    }
                    List<String> cats = new ArrayList<>(set);
                    cats.add("__UNKNOWN__");
                    categories.put(c, cats);
                }
            }
        }

        Encoded encode(Table t, ColumnType[] types) {
            List<String> featNames = new ArrayList<>();

            // Count output dimensionality
            int outDim = 0;
            for (int c = 0; c < t.nCols(); c++) {
                if (types[c] == ColumnType.NUMERIC) {
                    outDim++;
                    featNames.add(t.header[c]);
                } else {
                    List<String> cats = categories.get(c);
                    for (String cat : cats)
                        featNames.add(t.header[c] + "=" + cat);
                    outDim += cats.size();
                }
            }

            double[][] X = new double[t.nRows()][outDim];

            for (int r = 0; r < t.nRows(); r++) {
                int pos = 0;
                String[] row = t.rows.get(r);

                for (int c = 0; c < t.nCols(); c++) {
                    if (types[c] == ColumnType.NUMERIC) {
                        X[r][pos++] = Double.parseDouble(row[c].trim());
                    } else {
                        String v = row[c].trim();
                        List<String> cats = categories.get(c);
                        int idx = cats.indexOf(v);
                        if (idx < 0) idx = cats.indexOf("__UNKNOWN__");
                        X[r][pos + idx] = 1.0;
                        pos += cats.size();
                    }
                }
            }

            return new Encoded(X, featNames.toArray(new String[0]));
        }
    }

    private static class Encoded {
        double[][] X;
        String[] featureNames;
        Encoded(double[][] X, String[] f) { this.X = X; this.featureNames = f; }
    }

    // ============================================================
    // NORMALIZATION
    // ============================================================

    private static class Normalizer {
        double[] mean, std;

        Normalizer(double[][] X) {
            int N = X.length, D = X[0].length;
            mean = new double[D];
            std = new double[D];

            // Compute mean
            for (int j = 0; j < D; j++) {
                double s = 0;
                for (int i = 0; i < N; i++) s += X[i][j];
                mean[j] = s / N;
            }

            // Compute std
            for (int j = 0; j < D; j++) {
                double s = 0;
                for (int i = 0; i < N; i++) {
                    double d = X[i][j] - mean[j];
                    s += d * d;
                }
                std[j] = Math.sqrt(s / N);
                if (std[j] < 1e-12) std[j] = 1.0;
            }
        }

        void applyInPlace(double[][] X) {
            int N = X.length, D = X[0].length;
            for (int i = 0; i < N; i++)
                for (int j = 0; j < D; j++)
                    X[i][j] = (X[i][j] - mean[j]) / std[j];
        }
    }

    // ============================================================
    // TABLE + CSV UTILITIES
    // ============================================================

    private static class Table {
        String[] header;
        List<String[]> rows;

        Table(String[] header, List<String[]> rows) {
            this.header = header;
            this.rows = rows;
        }

        int nRows() { return rows.size(); }
        int nCols() { return header.length; }

        int colIndex(String name) {
            for (int i = 0; i < header.length; i++)
                if (header[i].equals(name)) return i;
            throw new RuntimeException("Column not found: " + name);
        }

        Table removeColumn(String name) {
            int c = colIndex(name);
            String[] newHeader = new String[header.length - 1];
            System.arraycopy(header, 0, newHeader, 0, c);
            System.arraycopy(header, c + 1, newHeader, c, header.length - c - 1);

            List<String[]> newRows = new ArrayList<>();
            for (String[] r : rows) {
                String[] nr = new String[r.length - 1];
                System.arraycopy(r, 0, nr, 0, c);
                System.arraycopy(r, c + 1, nr, c, r.length - c - 1);
                newRows.add(nr);
            }
            return new Table(newHeader, newRows);
        }

        Table subset(int[] idx) {
            List<String[]> newRows = new ArrayList<>();
            for (int i : idx) newRows.add(rows.get(i));
            return new Table(header, newRows);
        }
    }

    private static Table readCSV(String path) throws IOException {
        try (BufferedReader br = new BufferedReader(new FileReader(path))) {
            String headerLine = br.readLine();
            if (headerLine == null) throw new IOException("Empty CSV: " + path);

            String[] header = headerLine.split(",");
            List<String[]> rows = new ArrayList<>();

            String line;
            while ((line = br.readLine()) != null) {
                String[] toks = line.split(",");
                if (toks.length != header.length)
                    continue; // skip malformed rows
                rows.add(toks);
            }
            return new Table(header, rows);
        }
    }

    private static int[] shuffledIndices(int N, long seed) {
        int[] a = new int[N];
        for (int i = 0; i < N; i++) a[i] = i;
        Random r = new Random(seed);
        for (int i = N - 1; i > 0; i--) {
            int j = r.nextInt(i + 1);
            int tmp = a[i]; a[i] = a[j]; a[j] = tmp;
        }
        return a;
    }

    private static double[] subsetArray(double[] y, int[] idx) {
        double[] out = new double[idx.length];
        for (int i = 0; i < idx.length; i++) out[i] = y[idx[i]];
        return out;
    }
}


