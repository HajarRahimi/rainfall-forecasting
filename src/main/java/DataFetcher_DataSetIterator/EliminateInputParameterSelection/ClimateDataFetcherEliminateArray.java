package DataFetcher_DataSetIterator.EliminateInputParameterSelection;

import net.didion.jwnl.data.Exc;
import org.deeplearning4j.datasets.fetchers.BaseDataFetcher;
import org.nd4j.linalg.dataset.DataSet;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.io.IOUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.springframework.core.io.ClassPathResource;

/**
 * Created by Hajar Rahimi on 5/13/2016.
 */
public class ClimateDataFetcherEliminateArray extends BaseDataFetcher {

    private int num_outcomes;
    private int input_columes;
    private int num_example;
    private int batchSize;
    public int[] indexes;
    public String location;

    public ClimateDataFetcherEliminateArray() {
        numOutcomes = this.num_outcomes;
        inputColumns = input_columes;
        totalExamples = num_example;
    }

    public ClimateDataFetcherEliminateArray(int num_outcomes, int num_example, int batch, String address) {
        numOutcomes = num_outcomes;
        totalExamples = num_example;
        batchSize = batch;
        location = address;
    }

    public ClimateDataFetcherEliminateArray(int num_outcomes, int num_example, int batch, int[] idxs, String address) {
        numOutcomes = num_outcomes;
        totalExamples = num_example;
        batchSize = batch;
        location = address;
        setIndex(idxs);
    }

    public void setIndex(int[] idxs) {
        inputColumns = idxs.length;
        indexes = idxs;
    }

    @Override
    public void fetch(int numExamples) {

        int from = cursor;
        int to = cursor + numExamples;
        if (to > totalExamples)
            to = totalExamples;

        try {
            try {
                initializeCurrFromList(LoadData(from, to));//---
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
            cursor += numExamples;//---
        } catch (IOException e) {//---
            throw new IllegalStateException("Unable to load Data_3000.csv");//---
        }//---

    }//---

    List<String> lines = null;

//    private  List<String> getLines() throws IOException{
//        if (lines == null){
//            ClassPathResource resource = new ClassPathResource(location);
//            lines = IOUtils.readLines(resource.getInputStream());
//        }
//
//        return lines;
//    }


    private List<DataSet> LoadData(int from, int to) throws IOException, InterruptedException {   //---
        ClassPathResource resource = new ClassPathResource(location);
        List<String> lines = IOUtils.readLines(resource.getInputStream());
        List<DataSet> outputList = new ArrayList<>();//---
        INDArray ret = Nd4j.ones(Math.abs(to - from), indexes.length);// number of input
        double[][] outcomes = new double[lines.size()][numOutcomes];//number of output
        String line = null;
        String[] split = null;
        String outcome = null;
        double[] rowOutcome = null;
        int i = 0;
        int putCount = 0;
        try {
            putCount = 0;
            for (i = from; i < to; i++) {
                line = lines.get(i);
                split = line.split(",");
                addRow(ret, putCount++, split);
                outcome = split[split.length - 1];
                rowOutcome = new double[numOutcomes];// number of output
                rowOutcome[Integer.parseInt(outcome)] = 1;
                outcomes[i] = rowOutcome;
            }
        } catch (Exception ex) {
            throw ex;
        }
        try {
            for (i = 0; i < ret.rows(); i++) {
                DataSet add = new DataSet(ret.getRow(i), Nd4j.create(outcomes[from + i]));
                outputList.add(add);
            }
        } catch (Exception ex) {
            throw ex;
        }
        return outputList;
    }

    private void addRow(INDArray ret, int row, String[] line) {

        double[] vector = null;
//        int cnt = 0;
//        int ii = 0;
        try {
            vector = new double[indexes.length];//number of input
//            for(int i : indexes){
//                double d = Double.parseDouble(line[i]);
//                vector[cnt++] = d;
//                ii = i;
//            }

            for (int i = 0; i < (indexes.length); i++)//number of input
                vector[i] = Double.parseDouble(line[indexes[i]]);

            ret.putRow(row, Nd4j.create(vector));
        } catch (Exception ex) {
            throw ex;
        }

    }
}