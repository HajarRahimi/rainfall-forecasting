package DataFetcher_DataSetIterator.EliminateInputParameterSelection;

import org.deeplearning4j.datasets.iterator.BaseDatasetIterator;

/**
 * Created by Hajar Rahimi on 7/1/2016.
 */
public class ClimateDataSetIteratorEliminateArray extends BaseDatasetIterator {

    private  ClimateDataFetcherEliminateArray dataFetcherEliminateArray = null;
    public ClimateDataSetIteratorEliminateArray(int num_outcomes, int num_examples, int batch, int[] idxs, String address) {
        super(num_examples, batch, new ClimateDataFetcherEliminateArray(num_outcomes, num_examples, batch, idxs, address));

    }

    public ClimateDataSetIteratorEliminateArray(int num_outcomes, int num_examples, int batch, String address) {
        super(num_examples, batch,  new ClimateDataFetcherEliminateArray(num_outcomes, num_examples, batch, address));
    }

    public void setIndex(int[] idxs){
        ClimateDataFetcherEliminateArray df = (ClimateDataFetcherEliminateArray)(this.fetcher);
        df.setIndex(idxs);
    }


}
