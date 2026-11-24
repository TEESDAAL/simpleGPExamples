package iris_multi_tree_classification.simpleGP;

import gp.core.assessor.IndividualAssessor;
import gp.core.breeder.Breeder;
import gp.core.fitness.Goal;
import gp.core.fitness.SingleObjectiveFitness;
import gp.core.individual.AssessedIndividual;
import gp.core.initializer.Initialiser;
import gp.core.primitive_set.PrimitiveSet;
import gp.core.primitive_set.PrimitiveSetBuilder;
import gp.core.selector.Sampler;
import gp.core.statistic.Statistic;
import gp.impl.breeder.NaiveBreeder;
import gp.impl.genetic_operator.Identity;
import gp.impl.genetic_operator.SimpleGeneticOperator;
import gp.impl.genetic_operator.crossover.CommonNodeCounts;
import gp.impl.genetic_operator.crossover.ValidNodeCounter;
import gp.impl.genetic_operator.multitree.MSwapNCrossover;
import gp.impl.genetic_operator.multitree.NSubtreeMutation;
import gp.impl.individual.SingleTreeIndividual;
import gp.impl.individual.multitree.MultiTreeUtils;
import gp.impl.initializer.Initialisers;
import gp.impl.selector.Elitism;
import gp.impl.selector.TournamentSelection;
import gp.impl.selector.random.DistributionBuilder;
import gp.impl.statistic.CommonFunctions;
import gp.impl.statistic.NumericUtils;
import gp.impl.statistic.csv_utils.CSVEntriesBuilder;
import iris_multi_tree_classification.simpleGP.data_set.Features;
import iris_multi_tree_classification.simpleGP.data_set.IrisClass;
import iris_multi_tree_classification.simpleGP.data_set.IrisDataSet;
import util.Utils;
import util.dataset.DataSet;
import util.dataset.TrainValidationTestSplit;
import util.random.RandomSource;
import util.sequence.impls.Trio;

import java.nio.file.Path;
import java.util.Comparator;
import java.util.function.Function;

import static gp.impl.individual.tree.tree_utils.CommonTypeTokens.DOUBLE;

public class Params {
    final int seed;
    final RandomSource random;
    final int populationSize = 1000;
    final int numGens = 50;
    final int operatorMaxDepth = 10;
    final int operatorMaxAttempts = 1;
    final int initialisationMaxDepth = 10;
    final ValidNodeCounter<Features> validCrossoverNodes = CommonNodeCounts.allValid();

    final PrimitiveSet<Features> primitiveSet = PrimitiveSetBuilder.<Features, Double>withDefaultType(
                    DOUBLE)
            .addBinaryFunction("+", Double::sum)
            .addBinaryFunction("-", (a, b) -> a - b)
            .addBinaryFunction("*", (a, b) -> a * b)
            .addBinaryFunction("%", NumericUtils.SAFE_DIVISION)
            .addBinaryFunction("min", Math::min)
            .addBinaryFunction("min", Math::max)
            .addUnaryFunction("neg", a -> -a)
            .addTerminal("SEPAL_LENGTH", f -> f.sepalLength().cm())
            .addTerminal("SEPAL_WIDTH", f -> f.sepalWidth().cm())
            .addTerminal("PETAL_LENGTH", f -> f.petalLength().cm())
            .addTerminal("PETAL_WIDTH", f -> f.petalWidth().cm())
            .build();

    final Initialiser<SingleTreeIndividual<Features, Double>> singleTreeInitialiser;
    final Initialiser<FlowerPredictor> initialiser;
    final Sampler<SimpleGeneticOperator<FlowerPredictor>> geneticOperators;
    final Breeder<AssessedIndividual<
            Features, IrisClass,
            FlowerPredictor,
            SingleObjectiveFitness
            >, FlowerPredictor> breeder;

    final Function<IrisDataSet, IndividualAssessor<
            Features, IrisClass,
            FlowerPredictor,
            SingleObjectiveFitness
    >> assessorCreator = dataset -> ind -> SingleObjectiveFitness.of(
            dataset.zip(
                    (features, irisType) -> ind.evaluate(features)
                                            == irisType
            ).mapToInt(Utils::boolToInt).sum() / (double) dataset.size(),
            Goal.MAXIMIZE
    );

    final Function<Integer, Statistic<AssessedIndividual<
            Features, IrisClass,
            FlowerPredictor,
            SingleObjectiveFitness
            >, String>> csvRecorder;
    final IndividualAssessor<Features, IrisClass, FlowerPredictor, SingleObjectiveFitness> trainAssessor;
    final IndividualAssessor<Features, IrisClass, FlowerPredictor, SingleObjectiveFitness> validationAssessor;
    final IndividualAssessor<Features, IrisClass, FlowerPredictor, SingleObjectiveFitness> testAssessor;

    public Params(int seed) {
        this.seed = seed;
        this.random = RandomSource.of(seed);
        this.singleTreeInitialiser = Initialisers.rampedHalfAndHalf(
                initialisationMaxDepth,
                random,
                primitiveSet,
                populationSize,
                100,
                DOUBLE
        );
        this.initialiser = MultiTreeUtils.initialiseWith(
                singleTreeInitialiser,
                singleTreeInitialiser,
                singleTreeInitialiser,
                FlowerPredictor::new
        );
        this.geneticOperators = DistributionBuilder.<SimpleGeneticOperator<FlowerPredictor>>withDefault(
                        Identity.of((i, l) -> new FlowerPredictor(
                                i.first(),
                                i.second(),
                                i.third(),
                                l
                        ))
                )
                .addElement(
                        0.65,
                        new NSubtreeMutation<Features, Double, Trio<SingleTreeIndividual<?, ?>>, FlowerPredictor>(
                                random,
                                operatorMaxDepth,
                                operatorMaxAttempts,
                                1,
                                (trees, lineage) -> new FlowerPredictor(
                                        (SingleTreeIndividual<Features, Double>) trees.first(),
                                        (SingleTreeIndividual<Features, Double>) trees.second(),
                                        (SingleTreeIndividual<Features, Double>) trees.third(),
                                        lineage
                                ),
                                primitiveSet
                        )
                )
                .addElement(
                        0.20,
                        new MSwapNCrossover<>(
                                random,
                                operatorMaxDepth,
                                operatorMaxAttempts,
                                1,
                                1,
                                validCrossoverNodes,
                                (trees, lineage) -> new FlowerPredictor(
                                        (SingleTreeIndividual<Features, Double>) trees.first(),
                                        (SingleTreeIndividual<Features, Double>) trees.second(),
                                        (SingleTreeIndividual<Features, Double>) trees.third(),
                                        lineage
                                )
                        )
                ).toSampler(random);

        this.breeder = new NaiveBreeder<>(
                geneticOperators,
                populationSize,
                TournamentSelection.of(
                        random,
                        7,
                        SingleObjectiveFitness.directComparison()
                ),
                Elitism.of(
                        10,
                        SingleObjectiveFitness.directComparison()
                )
        );

        TrainValidationTestSplit<DataSet<Features, IrisClass>> splits = IrisDataSet.fromCSV(
                Path.of(
                        "/Users/alant/Documents/simpleGPExamples/src/main/resources/bezdekIris.data"
                )).get().trainValidationTestSplit(
                random,
                0.5,
                0.2,
                0.3
        );

        var trainingDataset = new IrisDataSet(splits.trainingSet().data());
        var validationDataset = new IrisDataSet(splits.validationSet().data());
        var testingDataset = new IrisDataSet(splits.testSet().data());

        this.trainAssessor = this.assessorCreator.apply(trainingDataset);
        this.validationAssessor = this.assessorCreator.apply(validationDataset);
        this.testAssessor = this.assessorCreator.apply(testingDataset);

        this.csvRecorder = CSVEntriesBuilder.<AssessedIndividual<
                        Features,
                        IrisClass,
                        FlowerPredictor,
                        SingleObjectiveFitness
        >>of(Path.of("out", "iris_" + seed + ".csv").toFile())
                .addStr("gen", (gen, _) -> gen)
                .addShared((_, pop) -> pop.stream().max(Comparator.comparingDouble(i -> i.fitness().score())).orElseThrow(),
                        builder -> builder
                                .addStr("bestFitness", i -> i.fitness().score())
                                .addStr("validationFitness", i -> validationAssessor.assessIndividual(i.individual()).score())
                                .addStr("sizeOfBestTree1", i -> i.individual().first().tree().size())
                                .addStr("sizeOfBestTree2", i -> i.individual().second().tree().size())
                                .addStr("sizeOfBestTree3", i -> i.individual().third().tree().size())
                )
                .<AssessedIndividual<?, ?, FlowerPredictor, SingleObjectiveFitness>, Number>withTemplate(
                        f -> (_, pop) -> pop.stream().map(f).collect(CommonFunctions.MEAN),
                        builder -> builder
                                .addTemplate("avgFitness", i -> i.fitness().score())
                                .addTemplate("avgSizeOfTree1", i -> i.individual().first().tree().size())
                                .addTemplate("avgSizeOfTree2", i -> i.individual().second().tree().size())
                                .addTemplate("avgSizeOfTree3", i -> i.individual().third().tree().size())
                )
                .build();
    }
}
