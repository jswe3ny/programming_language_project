package ml.helpers;

import java.io.*;
import java.nio.file.*;

/** Counts physical SLOC for a single .java file (ignores blanks & comment-only lines). */
public final class SLOC {
    private SLOC() {}

    /** Count SLOC for the source file that defines the given class.
     *  srcRoot: usually "src". Example: for ml.models.KNN -> src/ml/models/KNN.java
     */
    public static int forClass(Class<?> cls, String srcRoot) {
        String rel = cls.getName().replace('.', '/') + ".java";
        Path p = Paths.get(srcRoot, rel);
        return forFile(p);
    }

    /** Count SLOC for an explicit path. */
    public static int forFile(Path path) {
        if (!Files.exists(path)) return -1;
        int count = 0;
        boolean inBlock = false;
        try (BufferedReader br = Files.newBufferedReader(path)) {
            String line;
            while ((line = br.readLine()) != null) {
                String s = line;
                // Strip inline block comments while handling /* ... */ possibly spanning lines
                while (true) {
                    if (inBlock) {
                        int end = s.indexOf("*/");
                        if (end < 0) { s = ""; break; }
                        s = s.substring(end + 2);
                        inBlock = false;
                    } else {
                        int start = s.indexOf("/*");
                        int dbl = s.indexOf("//");
                        if (start >= 0 && (dbl < 0 || start < dbl)) {
                            int end = s.indexOf("*/", start + 2);
                            if (end >= 0) {
                                s = s.substring(0, start) + s.substring(end + 2);
                            } else {
                                s = s.substring(0, start);
                                inBlock = true;
                            }
                        } else break;
                    }
                }
                // Strip // comments
                int cmt = s.indexOf("//");
                if (cmt >= 0) s = s.substring(0, cmt);
                if (s.trim().length() > 0) count++;
            }
        } catch (IOException e) {
            return -1;
        }
        return count;
    }
}

