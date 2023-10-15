package jeneticsx;

import org.jenetics.Alterer;
import org.jenetics.Crossover;
import org.jenetics.Gene;
import org.jenetics.Population;
import org.jenetics.util.MSeq;

/**
 * Created by Mahdi on 7/27/2016.
 */
public class NNGenotypeCrossover <
        G extends Gene<?, G>,
        C extends Comparable<? super C>
        >
        extends Crossover<G, C>
{
    public  NNGenotypeCrossover(){
        super(.05);
    }

    @Override
    protected int crossover(MSeq<G> that, MSeq<G> other) {
        return 0;
    }
}