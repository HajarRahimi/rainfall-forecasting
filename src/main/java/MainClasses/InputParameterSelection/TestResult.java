package MainClasses.InputParameterSelection;

import DataFetcher_DataSetIterator.EliminateDBN.ClimateDataSetIteratorEliminateDBN;
import DataFetcher_DataSetIterator.EliminateInputParameterSelection.ClimateDataSetIteratorEliminateArray;
import MainClasses.DBN.DBN_Forecasting;
import MainClasses.EliminateInputParameterSelection.ClimatePredictor;
import MainClasses.EliminateInputParameterSelection.Result;
import MainClasses.Helper.JeneticsConvertor;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.Layer;
import org.jenetics.internal.util.model;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Created by Hajar Rahimi on 7/20/2016.
 */
public class TestResult {
    private static Logger log = LoggerFactory.getLogger(DBN_Forecasting.class);

    private static ClimatePredictor predictor = null;

    public static ClimatePredictor getPredictor() {
        if (predictor == null)
            predictor = new ClimatePredictor();
        return predictor;
    }

    public static void main(String[] args) {
        String address = "/weather_SpreadSubsample.csv";
        int num_outcomes = 5;
        int batch = 1720;
        int num_examples = 2610;
        int iterations = 10;
        int seed = 123;
        int listenerFreq = 1;
        int splitTrainNum = (int) (batch * .7);
        int [] idxs = {0,1,2,5,7,8,12,13,18,19,20,21,22,26,27,28,29,30,33,38,41,42,43,44,45,46,48,51,54,56,58,59,60,63,65,67,68,71,73,74,76,77,80,82,83,85,87,88,90,92,96,97,98,99,101,102,104,105,107,108,109,110,114,119,120,122,123,125,131};


        Result result = null;

        try {
            result = getPredictor().RunEliminateArray(idxs, num_outcomes, batch, num_examples, iterations, seed, listenerFreq, splitTrainNum, address);
//            System.out.println("Gt: \t" + gt.toString());
//            System.out.println("NewGt: \t" + StringHelper.ArrayToString(newgt));
            System.out.println("Indexes: " + JeneticsConvertor.IntArrayToString(idxs));
            System.out.println("F1Score: " + result.getF1Score());
            System.out.println("-----------------------------------------------------------------");

        } catch (Exception e) {

        }

    }

}
