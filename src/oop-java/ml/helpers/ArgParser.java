package ml.helpers;
import java.util.*;

public class ArgParser {
    private final Map<String,String> m = new HashMap<>();
    public ArgParser(String[] args){
        for (int i=0;i<args.length;i++){
            if(args[i].startsWith("--")){
                String k=args[i].substring(2);
                String v = (i+1<args.length && !args[i+1].startsWith("--")) ? args[++i] : "true";
                m.put(k, v);
            }
        }
    }

    public String get(String k, String def){ 
        return m.getOrDefault(k, def); 
    }
    public boolean has(String k){ 
        return m.containsKey(k); 
    }
    public int getInt(String k, int def){ 
        try{
            return Integer.parseInt(m.get(k));
        }catch(Exception e){
            return def;
        } 
    }
    public double getDouble(String k, double def){ 
        try{
            return Double.parseDouble(m.get(k));
        }catch(Exception e){
            return def;
        } 
    }
}