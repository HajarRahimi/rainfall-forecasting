package DataFetcher_DataSetIterator.EliminateDBN;

import org.apache.commons.io.IOUtils;
import org.deeplearning4j.datasets.fetchers.BaseDataFetcher;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.springframework.core.io.ClassPathResource;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by Hajar Rahimi on 6/15/2016.
 */
public class ClimateDataFetcherEliminateDBN extends BaseDataFetcher {

    private int numExamples;
    private int[] columnIndexes;
    private int[] output_Indexes;
    private  int batchSize;
    public String location;

    public ClimateDataFetcherEliminateDBN(int outputNum, int batch, int numExamples, int[] inputIndexes, int[] outputIndexes, String address) {
        numOutcomes = outputNum;
        totalExamples = numExamples;
        batchSize = batch;
        inputColumns = inputIndexes.length;
        columnIndexes = inputIndexes;
        output_Indexes = outputIndexes;
        batchSize = batch;
        location = address;
    }

    @Override
    public void fetch(int numExamples) {
        int from = cursor;
        int to = cursor + numExamples;
        if (to > totalExamples) {
            to = totalExamples;
        }

        try {
            try {
                List<DataSet> data = LoadData(from, to);
                initializeCurrFromList(data);//---
            } catch (InterruptedException e) {
                e.getMessage();
            }
            cursor += numExamples;//---
        } catch (IOException e) {//---
            throw new IllegalStateException("Unable to load Data_16536.csv");//---
        }//---

    }//---


    private List<DataSet> LoadData(int from, int to) throws IOException, InterruptedException {   //---

        ClassPathResource resource = new ClassPathResource(location);

        int Length = columnIndexes.length;


        List<String> lines = IOUtils.readLines(resource.getInputStream());
        List<DataSet> outputList = new ArrayList<>();//---
        INDArray ret = Nd4j.ones(Math.abs(to - from), Length);// number of input
        double[][] outcomes = new double[lines.size()][numOutcomes];//number of output
        int putCount = 0;
        for (int i = from; i < to; i++) {
            String line = lines.get(i);
            String[] split = line.split(",");

            addRow(ret, putCount++, split, Length, columnIndexes);

            String outcome = split[split.length - 1];
            double[] rowOutcome = new double[numOutcomes];// number of output
            rowOutcome[Integer.parseInt(outcome)] = 1;
            outcomes[i] = rowOutcome;
        }

        for (int i = 0; i < ret.rows(); i++) {
            DataSet add = new DataSet(ret.getRow(i), Nd4j.create(outcomes[from + i]));
            outputList.add(add);
        }
        return outputList;
    }

    private static void addRow(INDArray ret, int row, String[] line, int length, int[] columnIndexes) {
        double[] vector = new double[length];//number of input

        for (int i = 0; i < (length); i++)//number of input
                vector[i] = Double.parseDouble(line[columnIndexes[i]]);

        ret.putRow(row, Nd4j.create(vector));
    }
}
