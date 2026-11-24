package linear_regression.simpleGP;

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
import gp.core.statistic.StatisticBuilder;
import gp.impl.breeder.NaiveBreeder;
import gp.impl.genetic_operator.Identity;
import gp.impl.genetic_operator.SimpleGeneticOperator;
import gp.impl.genetic_operator.SingleTreeCrossover;
import gp.impl.genetic_operator.crossover.CommonNodeCounts;
import gp.impl.genetic_operator.crossover.ValidNodeCounter;
import gp.impl.genetic_operator.mutation.SingleTreeSubtreeMutation;
import gp.impl.individual.SingleTreeIndividual;
import gp.impl.initializer.Initialisers;
import gp.impl.selector.Elitism;
import gp.impl.selector.TournamentSelection;
import gp.impl.selector.random.DistributionBuilder;
import gp.impl.statistic.DoubleData;
import gp.impl.statistic.EvaluationFunctions;
import util.random.RandomSource;

import java.util.function.DoubleBinaryOperator;
import java.util.stream.IntStream;

import static gp.impl.individual.tree.tree_utils.CommonTypeTokens.DOUBLE;

public class Params {
    final int seed = 42;
    final int numGens = 50;
    final int populationSize = 1000;
    final int initialisationMaxDepth = 8;
    final int operatorMaxDepth = 10;
    final int operatorMaxAttempts = 10;
    final ValidNodeCounter<Point2D> validCrossoverNodes = CommonNodeCounts.allValid();
    final RandomSource random = RandomSource.of(seed);
    final int trainingPointsToSample = 10;

    final PrimitiveSet<Point2D> primitiveSet = PrimitiveSetBuilder.<Point2D, Double>withDefaultType(DOUBLE)
            .addBinaryFunction("add", Double::sum)
            .addBinaryFunction("sub", (a, b) -> a - b)
            .addBinaryFunction("mul", (a, b) -> a * b)
            .addTerminal("x", Point2D::x)
            .addTerminal("y", Point2D::y)
            .build();

    final Initialiser<SingleTreeIndividual<Point2D, Double>> initialiser = Initialisers.rampedHalfAndHalf(
            initialisationMaxDepth,
            random,
            primitiveSet,
            populationSize,
            100,
            DOUBLE
    );


    final Sampler<SimpleGeneticOperator<SingleTreeIndividual<Point2D, Double>>> geneticOperators =
            DistributionBuilder.<SimpleGeneticOperator<SingleTreeIndividual<Point2D, Double>>>withDefault(
                    Identity.of((i, l) -> SingleTreeIndividual.of(i.tree(), l))
            ).addElement(0.01, new SingleTreeSubtreeMutation<>(
                    random,
                    primitiveSet,
                    operatorMaxDepth,
                    operatorMaxAttempts
            )).addElement(0.65, new SingleTreeCrossover<>(
                    random,
                    operatorMaxAttempts,
                    operatorMaxDepth,
                    validCrossoverNodes
            )).toSampler(random);

    final Breeder<AssessedIndividual<
            Point2D, Double,
            SingleTreeIndividual<Point2D, Double>,
            SingleObjectiveFitness
    >, SingleTreeIndividual<Point2D, Double>> breeder = new NaiveBreeder<>(
            geneticOperators,
            populationSize,
            TournamentSelection.of(
                    random,
                    7,
                    SingleObjectiveFitness.directComparison()
            ),
            Elitism.of(10, SingleObjectiveFitness.directComparison())
    );

    DoubleBinaryOperator trueFunction
            = (x, y) -> (x * x * y) + (x * y + y);

    final IndividualAssessor<
            Point2D, Double,
            SingleTreeIndividual<Point2D, Double>,
            SingleObjectiveFitness
    > assessor = individual -> SingleObjectiveFitness.of(
            IntStream.range(0, trainingPointsToSample)
                    .mapToObj(_ -> new Point2D(
                            random.nextDouble(-Math.PI, Math.PI),
                            random.nextDouble(-Math.PI, Math.PI)
                    ))
                    .map(p -> DoubleData.of(
                            individual.evaluate(p),
                            p.reduce(trueFunction)
                    ))
                    .collect(EvaluationFunctions.SUM_ABSOLUTE_ERROR),
            Goal.MINIMIZE
    );

    final Statistic<AssessedIndividual<
            Point2D, Double,
            SingleTreeIndividual<Point2D, Double>,
            SingleObjectiveFitness
    >, String> statistics = StatisticBuilder.<
                    Point2D, Double, 
                    SingleTreeIndividual<Point2D, Double>,
                    SingleObjectiveFitness
    >separator(":", "\n")
            .ofFitnesses("BEST", fits -> fits.mapToDouble(SingleObjectiveFitness::score).min())
            .ofFitnesses("AVERAGE", fits -> fits.mapToDouble(SingleObjectiveFitness::score).average())
            .ofIndividuals("AVERAGE SIZE", inds -> inds.mapToInt(i -> i.tree().size()))
            .logTo(System.out::println)
            .build();
}
