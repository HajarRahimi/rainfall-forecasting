/**
 * C:\Users\Hajar Rahimi\Documents\GitHub\dl4j-0.4-examples\src\main\java\org\deeplearning4j\examples\mlp\sampleNetStructure\CMGSNet
 */

package MainClasses.EliminateInputParameterSelection;

import DataFetcher_DataSetIterator.EliminateInputParameterSelection.ClimateDataSetIteratorEliminateArray;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.RBM;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.jenetics.internal.util.model;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Random;

/**
 * Created by Hajar Rahimi.
 */

public class ClimatePredictor {
    private static Logger log = LoggerFactory.getLogger(ClimatePredictor.class);

//    private ClimateDataSetIteratorEliminateArray iteratorEliminateArray;
//
//    private ClimateDataSetIteratorEliminateArray getIterator(int num_outcomes, int num_examples, int batch, String address) {
//        if (iteratorEliminateArray == null)
//            iteratorEliminateArray = new ClimateDataSetIteratorEliminateArray(num_outcomes, batch, num_examples, address);
//        return iteratorEliminateArray;
//    }

    public static  Result RunEliminateArray(int[] idxs, int num_outcomes, int batch, int num_examples, int iterations, int seed, int listenerFreq, int splitTrainNum, String address) throws Exception {
        // Customizing params
        Nd4j.MAX_SLICES_TO_PRINT = -1;
        Nd4j.MAX_ELEMENTS_PER_SLICE = -1;
        ClimateDataSetIteratorEliminateArray iterator = null;
        DataSet next = null;
        try {
            iterator = new ClimateDataSetIteratorEliminateArray(num_outcomes, num_examples, num_examples, address);

            iterator.setIndex(idxs);
            next = iterator.next();

        } catch (Exception ex) {
            throw ex;
        }

        next.normalizeZeroMeanZeroUnitVariance();
        SplitTestAndTrain testAndTrain = next.splitTestAndTrain(splitTrainNum, new Random(seed));
        DataSet train = testAndTrain.getTrain();
        DataSet test = testAndTrain.getTest();
        Nd4j.ENFORCE_NUMERICAL_STABILITY = true;

        log.info("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerParamType)
                .updater(Updater.RMSPROP)
                .learningRate(1e-2) // TODO create learnable lr that shrinks by multiplicative constant after each epoch pg 3
                .list(3)
                .layer(0, new RBM.Builder()
                        .nIn(idxs.length)
                        .nOut(16)
                        .weightInit(WeightInit.UNIFORM)
                        .lossFunction(LossFunctions.LossFunction.MCXENT)
                        .hiddenUnit(RBM.HiddenUnit.RECTIFIED)
                        .visibleUnit(RBM.VisibleUnit.SOFTMAX)
                        .activation("relu")
                        .build())
                .layer(1, new RBM.Builder()
                        .nIn(16)
                        .nOut(17)
                        .weightInit(WeightInit.UNIFORM)
                        .lossFunction(LossFunctions.LossFunction.MCXENT)
                        .hiddenUnit(RBM.HiddenUnit.RECTIFIED)
                        .visibleUnit(RBM.VisibleUnit.SOFTMAX)
                        .activation("relu")
                        .dropOut(0.5)
                        .build())
//                .layer(2, new RBM.Builder()
//                        .nIn(2)
//                        .nOut(23)
//                        .weightInit(WeightInit.UNIFORM)
//                        .lossFunction(LossFunctions.LossFunction.MCXENT)
//                        .hiddenUnit(RBM.HiddenUnit.RECTIFIED)
//                        .visibleUnit(RBM.VisibleUnit.SOFTMAX)
//                        .activation("relu")
//                        .dropOut(0.5)
//                        .build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .nIn(17)
                        .nOut(num_outcomes)
                        .activation("softmax")
                        .weightInit(WeightInit.UNIFORM)
                        .build())
                .backprop(true)
                .pretrain(false)
                .build();
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        model.setListeners(new ScoreIterationListener(listenerFreq));
//        log.info("Train model....");
        model.fit(train);

//        log.info("Evaluate weights....");
        for (Layer layer : model.getLayers()) {
            INDArray w = layer.getParam(DefaultParamInitializer.WEIGHT_KEY);
//            log.info("Weights: " + w);
        }

//        log.info("Evaluate model....");
        Evaluation eval = new Evaluation(num_outcomes);
        INDArray output = model.output(test.getFeatureMatrix(), Layer.TrainingMode.TEST);

//        for (int i = 0; i < output.rows(); i++) {
//            String actual = test.getLabels().getRow(i).toString().trim();
//            String predicted = output.getRow(i).toString().trim();
//            log.info("actual " + actual + " vs predicted " + predicted);
//        }

        eval.eval(test.getLabels(), model.output(test.getFeatureMatrix(), Layer.TrainingMode.TEST));
//        log.info(eval.stats());

//        log.info("****************Example finished********************");

        double acc = eval.accuracy();
        double prec = eval.precision();
        double reca = eval.recall();
        double f1 = eval.f1();
        //System.out.println(Thread.currentThread().getId() + " f1: " + f1);

        Result result = new Result(acc, prec, reca, f1);

        return result;
    }
}