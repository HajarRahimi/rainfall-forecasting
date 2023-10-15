package MainClasses.InputParameterSelection;

import MainClasses.EliminateInputParameterSelection.ClimatePredictor;
import MainClasses.EliminateInputParameterSelection.Result;
import MainClasses.Helper.JeneticsConvertor;
import org.jenetics.*;
import org.jenetics.engine.Engine;
import org.jenetics.engine.EvolutionStatistics;

import java.util.ArrayList;
import java.util.Date;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import static org.jenetics.engine.EvolutionResult.toBestPhenotype;
import static org.jenetics.engine.limit.bySteadyFitness;

/**
 * Created by Hajar Rahimi
 */
public class GeneticParameterSelector {
//    static int num_outcomes;
//    static int batch;
//    static int num_examples;
//    static int iterations;
//    static int seed;
//    static int listenerFreq;
//    static int splitTrainNum;
//    static String address;

//    private static   ClimatePredictor predictor = null;
//    public static ClimatePredictor getPredictor(){
//        if (predictor == null)
//            predictor = new ClimatePredictor();
//        return  predictor;
//    }


    // 2.) Definition of the fitness function.
    private static  double eval(Genotype<BitGene> gt) {

        String address = "/Data_Split_Average_3000.csv";
        int num_outcomes = 5;
        int batch = 1720;
        int num_examples = 2610;
        int iterations = 10;
        int seed = 123;
        int listenerFreq = 1;
        int splitTrainNum = (int) (batch * .7);
        long threadId = Thread.currentThread().getId();
        int[] idxs = JeneticsConvertor.BitChromosomeToIntArray((BitChromosome) gt.getChromosome());
        try {
            StringBuilder sb = new StringBuilder();
            sb.append("--------------------------------------------------------------------\n");
            sb.append(threadId + " IN EVAL"+ "\n");
            //ClimatePredictor predictor = new ClimatePredictor();
            Result result = ClimatePredictor.RunEliminateArray(idxs, num_outcomes, batch, num_examples, iterations, seed, listenerFreq, splitTrainNum, address);
            sb.append(threadId + " GT: " + JeneticsConvertor.BitChromosomeToString((BitChromosome) gt.getChromosome())+ "\n");
            sb.append(threadId + " Indexes: " + JeneticsConvertor.BitChromosomeToIntArrayToString((BitChromosome) gt.getChromosome())+ "\n");
            sb.append(threadId + " F1Score: " + result.getF1Score() + "\n");
            sb.append("-----------------------------------------------------------------\n");
            System.out.print(sb.toString());
            return result.getF1Score();
        } catch (Exception ex) {
            System.out.println("Eval Exception:" + ex.getMessage());
        }
        return 0;
    }

    public static void main(String[] args) {


        Date startTime = new Date();
//        num_outcomes = num_outcomes_param;
//        batch = batch_param;
//        num_examples = num_examples_param;
//        iterations = iterations_param;
//        seed = seed_param;
//        listenerFreq = listenerFreq_param;
//        splitTrainNum = splitTrainNum_param;
//        address = address_param;

        final ExecutorService executorService = Executors.newFixedThreadPool(1);

        // Configure and build the evolution engine.
        final Engine<BitGene, Double> engine = Engine
                .builder(
                        GeneticParameterSelector::eval,
                        BitChromosome.of(134, 0.5))
                .populationSize(200)
                .optimize(Optimize.MAXIMUM)
                .selector(new RouletteWheelSelector<>())
                .alterers(
                        new Mutator<>(0.15),
                        new SinglePointCrossover<>(0.35))
                .executor(executorService)
                .build();


        // Create evolution statistics consumer.
        final EvolutionStatistics<Double, ?>
                statistics = EvolutionStatistics.ofNumber();

        final Phenotype<BitGene, Double> best = engine.stream()
                // Truncate the evolution stream after 7 "steady"
                // generations.
                .limit(bySteadyFitness(30))
                // The evolution will stop after maximal 100
                // generations.
                .limit(300)
                // Update the evaluation statistics after
                // each generation
                .peek(statistics)
                // Collect (reduce) the evolution stream to
                // its best phenotype.
                .collect(toBestPhenotype());

        System.out.println(statistics);
        System.out.println("*********************************************************");
        System.out.println("Best chromosome (boolean): ");
        System.out.println(JeneticsConvertor.BitChromosomeToString((BitChromosome) best.getGenotype().getChromosome()));
        System.out.println("Best chromosome (indexes): ");
        System.out.println(JeneticsConvertor.BitChromosomeToIntArrayToString((BitChromosome) best.getGenotype().getChromosome()));
        System.out.println("Best Fitness: " + best.getFitness());
        System.out.println("*********************************************************");
        System.out.println("Start time: " + startTime.toString());
        System.out.println("End time: " + new Date().toString());
    }
}