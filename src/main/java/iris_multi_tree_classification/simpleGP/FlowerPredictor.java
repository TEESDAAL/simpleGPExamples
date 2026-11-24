package iris_multi_tree_classification.simpleGP;

import gp.core.individual.Individual;
import gp.impl.individual.SingleTreeIndividual;
import gp.impl.individual.lineage.Lineage;
import gp.impl.individual.multitree.fixed_size.ThreeTreeIndividual;
import iris_multi_tree_classification.simpleGP.data_set.Features;
import iris_multi_tree_classification.simpleGP.data_set.IrisClass;

import java.util.Comparator;
import java.util.stream.Stream;

public record FlowerPredictor(
        SingleTreeIndividual<Features, Double> setosaPredictor,
        SingleTreeIndividual<Features, Double> versicolourPredictor,
        SingleTreeIndividual<Features, Double> virginicaPredictor,
        Lineage lineage
) implements ThreeTreeIndividual<
        Features, Double,
        Features, Double,
        Features, Double,
        FlowerPredictor
>, Individual<Features, IrisClass> {
    @Override
    public SingleTreeIndividual<Features, Double> first() {
        return setosaPredictor;
    }

    @Override
    public SingleTreeIndividual<Features, Double> second() {
        return versicolourPredictor;
    }

    @Override
    public SingleTreeIndividual<Features, Double> third() {
        return virginicaPredictor;
    }

    record Prediction(double confidence, IrisClass prediction) {}

    /**
     * Classify the feature set by picking the tree with the highest confidence.
     * @param features The features of the flower.
     * @return The {@link IrisClass} with the highest confidence.
     */
    @Override
    public IrisClass evaluate(Features features) {
        return Stream.of(
                new Prediction(setosaPredictor.evaluate(features), IrisClass.SETOSA),
                new Prediction(versicolourPredictor.evaluate(features), IrisClass.VERSICOLOUR),
                new Prediction(virginicaPredictor.evaluate(features), IrisClass.VIRGINICA)
        ).max(Comparator.comparingDouble(Prediction::confidence))
                .orElseThrow()
                .prediction();
    }
}
