package DataFetcher_DataSetIterator.EliminateDBN;

import org.deeplearning4j.datasets.iterator.BaseDatasetIterator;

/**
 * Created by Hajar Rahimi on 5/13/2016.
 */
public class ClimateDataSetIteratorEliminateDBN extends BaseDatasetIterator {
        public ClimateDataSetIteratorEliminateDBN(int outputNum, int batch, int numExamples, int[] columnIndexes, int[] outputIndexes, String address){
        super(batch,numExamples,new ClimateDataFetcherEliminateDBN(outputNum, batch, numExamples, columnIndexes, outputIndexes, address));
    }
}
