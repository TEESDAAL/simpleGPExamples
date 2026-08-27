package complex_linear_regression.simpleGP;

import gp.core.assessor.IndividualAssessor;
import gp.core.breeder.Breeder;
import gp.core.fitness.Goal;
import gp.core.fitness.SingleObjectiveFitness;
import gp.core.individual.AssessedIndividual;
import gp.core.initializer.Initialiser;
import gp.core.initializer.PrimitiveSet;
import gp.core.initializer.PrimitiveSetBuilder;
import gp.core.selector.Sampler;
import gp.core.statistic.Statistic;
import gp.impl.breeder.NaiveBreeder;
import gp.impl.genetic_operators.CrossOver;
import gp.impl.genetic_operators.Identity;
import gp.impl.genetic_operators.SubtreeMutation;
import gp.impl.individual.SingleTreeIndividual;
import gp.impl.initializers.Initializers;
import gp.impl.selectors.Elitism;
import gp.impl.selectors.TournamentSelection;
import gp.impl.selectors.random.ProbabilisticElement;
import gp.impl.selectors.random.WeightedRandomSampler;
import gp.impl.statistic.EvaluationFunctions;
import gp.impl.statistic.NumericUtils;
import utils.Pair;
import utils.operators.Operator;
import utils.random.RandomSource;

import java.util.Arrays;
import java.util.List;
import java.util.function.BiFunction;
import java.util.stream.IntStream;

public class Params {
    final RandomSource random = RandomSource.of(42);
    final int populationSize = 1000;
    final int numGens = 50;

    final PrimitiveSet<Pair<Double, Double>> primitiveSet = PrimitiveSetBuilder.<Pair<Double, Double>>empty()
        .addNonTerminal("+", Double::sum, Double.class, Double.class, Double.class)
        .addNonTerminal("-", (a, b) -> a - b, Double.class, Double.class, Double.class)
        .addNonTerminal("*", (a, b) -> a * b, Double.class, Double.class, Double.class)
        .addNonTerminal("%", NumericUtils.SAFE_DIVISION, Double.class, Double.class, Double.class)
        .addNonTerminal("min", Math::min, Double.class, Double.class, Double.class)
        .addNonTerminal("min", Math::max, Double.class, Double.class, Double.class)
        .addNonTerminal("neg", a -> -a, Double.class, Double.class)
        .addTerminal("x", Pair::first, Double.class)
        .addTerminal("y", Pair::second, Double.class)
        .build();

    final Initialiser<SingleTreeIndividual<Pair<Double, Double>, Double>> initialiser = Initializers.rampedHalfAndHalf(
        8,
        random,
        primitiveSet,
        populationSize,
        100,
        Double.class
    );

    final Sampler<Operator<
        SingleTreeIndividual<Pair<Double, Double>, Double>,
        List<SingleTreeIndividual<Pair<Double, Double>, Double>>
        >> geneticOperators = WeightedRandomSampler.of(
        random,
        new Identity<>(),
        ProbabilisticElement.of(0.01, SingleTreeIndividual.operator(new SubtreeMutation<>(
            random, primitiveSet,
            17, 1
        ))),
        ProbabilisticElement.of(0.9, SingleTreeIndividual.operator(new CrossOver<>(random, 17, 1)))
    );

    final Breeder<AssessedIndividual<
        Pair<Double, Double>, Double,
        SingleTreeIndividual<Pair<Double, Double>, Double>,
        SingleObjectiveFitness
        >, SingleTreeIndividual<Pair<Double, Double>, Double>> breeder = new NaiveBreeder<>(
        geneticOperators,
        populationSize,
        TournamentSelection.of(random, 7, SingleObjectiveFitness.directlyCompare()),
        Elitism.of(10, SingleObjectiveFitness.directlyCompare())
    );

    BiFunction<Double, Double, Double> trueFunction = (x, y) -> Math.cos(y) * Math.exp(Math.sin(x));

    final IndividualAssessor<
        Pair<Double, Double>, Double,
        SingleTreeIndividual<Pair<Double, Double>, Double>,
        SingleObjectiveFitness
    > assessor = new IndividualAssessor<>() {
        @Override
        public SingleObjectiveFitness evaluate(SingleTreeIndividual<Pair<Double, Double>, Double> individual) {
            return SingleObjectiveFitness.of(
                IntStream.range(0, 100)
                    .mapToObj(_ -> Pair.of(
                        random.nextDouble(-Math.PI, Math.PI),
                        random.nextDouble(-Math.PI, Math.PI)
                    )).map(p -> Pair.of(individual.evaluate(p), p.reduce(trueFunction)))
                    .mapToDouble(p -> p.reduce((a, b) -> Math.abs(a-b)))
                    .sum(),
                Goal.MINIMIZE
            );
        }

        @Override
        public boolean shouldParallelize() {
            return false;
        }
    };

    final Statistic<AssessedIndividual<
        Pair<Double, Double>, Double,
        SingleTreeIndividual<Pair<Double, Double>, Double>,
        SingleObjectiveFitness
    >, String> statistics = pop -> {
        double[] fitnesses = pop.stream().mapToDouble(i -> i.fitness().score()).toArray();
        return "BEST: " + Arrays.stream(fitnesses).min().orElse(-1) + " AVG POP SIZE: " + pop.stream().mapToInt(i->i.individual().tree().size()).average().orElse(0.0);
    };


}
