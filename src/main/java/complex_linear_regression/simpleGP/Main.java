package complex_linear_regression.simpleGP;

import gp.GPPipeline;
import gp.TerminationCriterion;
import gp.core.statistic.SideEffect;


public class Main {
    static void main() {
        Params params = new Params();
        var _ = GPPipeline
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
