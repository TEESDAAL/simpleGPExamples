package complex_linear_regression.simpleGP;

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
import gp.impl.genetic_operator.mutation.SingleTreeSubtreeMutation;
import gp.impl.individual.SingleTreeIndividual;
import gp.impl.initializer.Initialisers;
import gp.impl.selector.Elitism;
import gp.impl.selector.TournamentSelection;
import gp.impl.selector.random.ProbabilisticElement;
import gp.impl.selector.random.WeightedRandomSampler;
import gp.impl.statistic.DoubleData;
import gp.impl.statistic.EvaluationFunctions;
import gp.impl.statistic.NumericUtils;
import util.random.RandomSource;

import java.util.function.BiFunction;
import java.util.stream.IntStream;

import static gp.impl.individual.tree.tree_utils.CommonTypeTokens.DOUBLE;

record Point2D(double x, double y) {}

public class Params {
    final RandomSource random = RandomSource.of(42);
    final int populationSize = 1000;
    final int numGens = 50;

    final PrimitiveSet<Point2D> primitiveSet = PrimitiveSetBuilder
            .<Point2D, Double>withDefaultType(DOUBLE)
            .addBinaryFunction("+", Double::sum)
            .addBinaryFunction("-", (a, b) -> a - b)
            .addBinaryFunction("*", (a, b) -> a * b)
            .addBinaryFunction("%", NumericUtils.SAFE_DIVISION)
            .addBinaryFunction("min", Math::min)
            .addBinaryFunction("min", Math::max)
            .addUnaryFunction("neg", a -> -a)
            .addTerminal("x", Point2D::x)
            .addTerminal("y", Point2D::y)
            .build();

    final Initialiser<SingleTreeIndividual<Point2D, Double>> initialiser = Initialisers.rampedHalfAndHalf(
            8,
            random,
            primitiveSet,
            populationSize,
            100,
            DOUBLE
    );

    final Sampler<SimpleGeneticOperator<
            SingleTreeIndividual<Point2D, Double>
    >> geneticOperators = WeightedRandomSampler.of(
            random,
            Identity.of(
                    (i, l) -> SingleTreeIndividual.of(i.tree(), l)
            ),
            ProbabilisticElement.of(
                    0.01, new SingleTreeSubtreeMutation<>(
                            random,
                            primitiveSet,
                            17,
                            1
                    )
            ),
            ProbabilisticElement.of(
                    0.9,
                    new SingleTreeCrossover<>(
                            random,
                            17,
                            10,
                            CommonNodeCounts.excludeRoot()
                    )
            )
    );

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

    BiFunction<Double, Double, Double> trueFunction =
            (x, y) -> Math.cos(y) * Math.exp(Math.sin(x));

    final IndividualAssessor<
            Point2D, Double,
            SingleTreeIndividual<Point2D, Double>,
            SingleObjectiveFitness
    > assessor = new IndividualAssessor<>() {
        @Override
        public SingleObjectiveFitness assessIndividual(
                SingleTreeIndividual<Point2D, Double> individual
        ) {
            return SingleObjectiveFitness.of(
                    IntStream.range(0, 100)
                            .mapToObj(_ -> new Point2D(
                                    random.nextDouble(
                                            -Math.PI,
                                            Math.PI
                                    ),
                                    random.nextDouble(
                                            -Math.PI,
                                            Math.PI
                                    )
                            )).map(p -> DoubleData.of(
                                    individual.evaluate(p),
                                    trueFunction.apply(p.x(), p.y())
                            ))
                            .collect(EvaluationFunctions.SUM_ABSOLUTE_ERROR),
                    Goal.MINIMIZE
            );
        }

        @Override
        public boolean shouldParallelize() {
            return false;
        }
    };

    final Statistic<AssessedIndividual<
            Point2D, Double,
            SingleTreeIndividual<Point2D, Double>,
            SingleObjectiveFitness
    >, String> statistics = StatisticBuilder.<
                    Point2D, Double,
                    SingleTreeIndividual<Point2D, Double>,
                    SingleObjectiveFitness
            >separator(": ", "\n")
            .ofFitnesses(
                    "BEST",
                    fit -> fit
                            .mapToDouble(SingleObjectiveFitness::score)
                            .min()
            )
            .ofIndividuals(
                    "AVG POP SIZE",
                    inds -> inds
                            .mapToInt(i -> i.tree().size())
                            .average()
            )
            .build();
}
