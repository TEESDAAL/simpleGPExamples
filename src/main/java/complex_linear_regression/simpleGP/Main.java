package complex_linear_regression.simpleGP;

import gp.GPPipeLine;
import gp.TerminationCriterion;
import gp.core.assessor.IndividualAssessor;
import gp.core.breeder.Breeder;
import gp.core.fitness.Goal;
import gp.core.fitness.SingleObjectiveFitness;
import gp.core.individual.AssessedIndividual;
import gp.core.initializer.Initialiser;
import gp.core.initializer.PrimitiveSet;
import gp.core.initializer.PrimitiveSetBuilder;
import gp.core.selector.Sampler;
import gp.core.statistic.SideEffect;
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
import utils.operators.SafeDivision;
import utils.random.RandomSource;

import java.util.Arrays;
import java.util.List;
import java.util.function.BiFunction;
import java.util.stream.IntStream;

public class Main {
    static void main() {
        Params params = new Params();
        GPPipeLine
            .start(params.initialiser::initialize)
            .repeat(TerminationCriterion.nIters(params.numGens),
                (i, pop) -> pop
                    .then(SideEffect.of(_ -> System.out.println(
                        "Evaluating population for gen " + i
                    )))
                    .then(params.assessor::assess)
                    .then(params.statistics)
                    .then(params.breeder::breed)
            )
            .then(params.assessor::assess)
            .then(params.statistics)
            .finish();
    }
}

