package MainClasses.Helper;

import org.jenetics.BitChromosome;
import org.jenetics.BitGene;

import java.util.ArrayList;

/**
 * Created by Mahdi on 7/29/2016.
 */
public class JeneticsConvertor {

    public static int[] BitChromosomeToIntArray(BitChromosome chromosome){
        ArrayList<Integer> list = new ArrayList<>();
        int i = 0;
        for (BitGene gene :
                chromosome) {
            if(gene.getBit())
                list.add(i);
            i++;
        }
        if (list.size() == 0)
            return  null;

        i = 0;
        int [] indexes = new int[list.size()];
        for (int index :
                list) {
            indexes[i] = index;
            i++;
        }
        return  indexes;
    }

    public static String BitChromosomeToString(BitChromosome chromosome){
        StringBuilder sb = new StringBuilder();
        sb.append("[");
        int cnt = 0;
        for (BitGene gene :
                chromosome) {
            sb.append(gene.getBit()? 1: 0);
            cnt++;
            if (cnt != chromosome.bitCount())
                sb.append(",");
        }
        sb.append("]");
        return  sb.toString();
    }

    public static String BitChromosomeToIntArrayToString(BitChromosome chromosome){
        int [] indexes = BitChromosomeToIntArray(chromosome);
        if (indexes == null)
            return "";
        return IntArrayToString(indexes);
    }


    public static String IntArrayToString(int [] array){
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
}
