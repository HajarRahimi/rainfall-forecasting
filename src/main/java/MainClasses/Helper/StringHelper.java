package MainClasses.Helper;


import org.jenetics.BitGene;
import org.jenetics.Phenotype;

/**
 * Created by Mahdi on 7/9/2016.
 */
public class StringHelper {
    public static String ArrayToString(int [] array){
        StringBuilder sb = new StringBuilder();
        sb.append("[");
        int cnt = 0;
        for (int i: array             ) {
            sb.append(i);
            cnt++;
            if (cnt != array.length)
            sb.append(",");
        }
        sb.append("]");
        return  sb.toString();
    }
    public static String PhenotypeToString(String best) {
        StringBuilder sb = new StringBuilder();
        sb.append("[");
        int cnt = 0;
        int elemAt = 0;
        int length = best.length();
        while (cnt < length) {
            if (best.charAt(length - 1 - cnt) == '1') {
                sb.append(elemAt++);
                sb.append(",");
                cnt++;
            } else if (best.charAt(length - 1 - cnt) == '0') {
                elemAt++;
                cnt++;
            } else {
                cnt++;
                continue;
            }
        }
        sb.deleteCharAt(sb.length()-1);
//        sb.deleteCharAt(sb.length()-1);
        sb.append("]");
        return sb.toString();
    }
}
