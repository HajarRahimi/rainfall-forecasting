package MainClasses.DBN;


import DataFetcher_DataSetIterator.EliminateDBN.ClimateDataSetIteratorEliminateDBN;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.RBM;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.weights.HistogramIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.Random;


/**
 * Created by agibsonccc on 9/12/14.
 */
public class Shallow_Neural_Network {

    private static Logger log = LoggerFactory.getLogger(Shallow_Neural_Network.class);

    public static void main(String[] args) throws IOException {
        // Customizing params
        Nd4j.MAX_SLICES_TO_PRINT = 10;
        Nd4j.MAX_ELEMENTS_PER_SLICE = 10;

        int[] columnIndexes = {1,2,3,4,5,9,10,12,13,17,18,21,26,27,30,32,33,38,43,44,48,51,55,56,59,60,61,62,63,65,67,68,70,73,74,76,78,82,84,87,93,94,98,100,103,104,112,114,123,124,131};
        int[] outputIndexes = {137};

        final int numRows = columnIndexes.length;
        final int numColumns = 1;
        int outputNum = 5;
        int numExamples = 16536;
        int batchSize = 16536;
        int iterations = 28;
        int splitTrainNum = (int) (batchSize * .69);
        int seed = 123;
        int listenerFreq = 1;
        String address = "/Data_Split_Average.csv";

        log.info("Load data....");
        ClimateDataSetIteratorEliminateDBN iter = new ClimateDataSetIteratorEliminateDBN(outputNum, batchSize, numExamples, columnIndexes, outputIndexes, address);
        DataSet next = iter.next();
//        System.out.println(next);
//        next.shuffle();
        next.normalizeZeroMeanZeroUnitVariance();


        log.info("Split data....");
        SplitTestAndTrain testAndTrain = next.splitTestAndTrain(splitTrainNum, new Random(seed));
        DataSet train = testAndTrain.getTrain();
        DataSet test = testAndTrain.getTest();
        Nd4j.ENFORCE_NUMERICAL_STABILITY = true;

        log.info("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .miniBatch(true)
                .seed(seed)
                .iterations(iterations)
                .optimizationAlgo(OptimizationAlgorithm.LBFGS)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerParamType)
                .updater(Updater.RMSPROP)
                .learningRate(1e-1) // TODO create learnable lr that shrinks by multiplicative constant after each epoch pg 3
                .list(4)
                .layer(0, new DenseLayer.Builder().nIn(numRows * numColumns).nOut(7)
                        .activation("relu")
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .layer(1, new DenseLayer.Builder()
                        .nIn(7)
                        .nOut(3)
                        .activation("relu")
                        .weightInit(WeightInit.NORMALIZED)
                        .build())
                .layer(2, new DenseLayer.Builder()
                        .nIn(3)
                        .nOut(8)
                        .activation("relu")
                        .weightInit(WeightInit.NORMALIZED)
                        .build())
                .layer(3, new OutputLayer.Builder(LossFunction.MCXENT)
                        .weightInit(WeightInit. NORMALIZED)
                        .activation("softmax")
                        .nIn(8)
                        .nOut(outputNum)
                        .build())
                .backprop(true)
                .pretrain(false)
                .build();
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        model.setListeners(new ScoreIterationListener(listenerFreq));
        log.info("Train model....");
        model.fit(train);

        log.info("Evaluate weights....");
        for (org.deeplearning4j.nn.api.Layer layer : model.getLayers()) {
            INDArray w = layer.getParam(DefaultParamInitializer.WEIGHT_KEY);
//            log.info("Weights: " + w);
        }

        log.info("Evaluate model....");
        Evaluation eval_1 = new Evaluation(outputNum);
        INDArray output_1 = model.output(test.getFeatureMatrix(), Layer.TrainingMode.TEST);

//        for (int i = 0; i < output_1.rows(); i++) {
//            String actual = test.getLabels().getRow(i).toString().trim();
//            String predicted = output_1.getRow(i).toString().trim();
//            log.info("actual " + actual + " vs predicted " + predicted);
//        }

        eval_1.eval(test.getLabels(), model.output(test.getFeatureMatrix(), Layer.TrainingMode.TEST));
        log.info(eval_1.stats());

        log.info("****************Train & Test finished********************");


        log.info("**********************Forecasting************************");

        log.info("Load forecast data....");
        numExamples = 840;
        batchSize = 840;
        address = "/forecasting_Average.csv";
        ClimateDataSetIteratorEliminateDBN iter_forecast = new ClimateDataSetIteratorEliminateDBN(outputNum, batchSize, numExamples, columnIndexes, outputIndexes, address);
        DataSet next_forecast = iter_forecast.next();
//        System.out.println(next_forecast);
        next_forecast.normalizeZeroMeanZeroUnitVariance();


        log.info("Forecasting model....");
        Evaluation eval_2 = new Evaluation(outputNum);

        INDArray output_2 = model.output(next_forecast.getFeatureMatrix(), Layer.TrainingMode.TEST);

        for (int i = 0; i < output_2.rows(); i++) {
            String actual = next_forecast.getLabels().getRow(i).toString().trim();
            String predicted = output_2.getRow(i).toString().trim();
            log.info("actual " + actual + " vs predicted " + predicted);
        }

        String output = "";
        double max = 0;
        int k = -1;
        for (int i = 0; i < output_2.rows(); i++) {
            INDArray predicted = output_2.getRow(i);
            for (int j = 0; j <= 4; j++) {
                if (max <= predicted.getDouble(j)) {
                    max = predicted.getDouble(j);
                    k = j;
                }
            }
            output += k;
            output += "\n";
            max = 0;
        }

        eval_2.eval(next_forecast.getLabels(), model.output(next_forecast.getFeatureMatrix(), Layer.TrainingMode.TEST));
        log.info(eval_2.stats());

        log.info("****************Example finished********************");

        try {

            FileWriter writer = new FileWriter("forecast_column_SallowNN.csv");
            writer.append(output);
            writer.flush();
            writer.close();
            System.exit(0);
        } catch (IOException e) {
            e.printStackTrace();
        }
        System.exit(0);
    }
}