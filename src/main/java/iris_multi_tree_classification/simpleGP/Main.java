package iris_multi_tree_classification.simpleGP;


import gp.GPPipeline;
import gp.TerminationCriterion;
import gp.core.statistic.SideEffect;
import util.Parallelizeable;
import util.result.Err;
import util.result.Ok;
import util.result.Result;

import java.util.stream.IntStream;

public class Main {
    static void main() {
        int numRuns = 1;
        Parallelizeable.of(3).parallelize(
                IntStream.range(0, numRuns).boxed(),
                Main::trainingRun
        ).forEach(System.out::println);
    }
    static RunResult trainingRun(Integer seed) {
        Params params = new Params(seed);
         return switch (Result.fromFunction(() -> GPPipeline
                 .start(params.initialiser::initialize)
                 .repeat(TerminationCriterion.nIters(params.numGens),
                         (i, pop) -> pop
                        .then(SideEffect.of(_ -> System.out.println(
                            "Evaluating population for gen " + i
                        )))
                        .then(params.trainAssessor::assess)
                        .then(params.csvRecorder.apply(i))
                        .then(params.breeder::breed)
                )
                .then(params.testAssessor::assess)

        //            .then(params.statistics)
                .finish())) {
             case Err(Throwable err) -> RunResult.FAILURE;
             case Ok(_) -> RunResult.SUCCESS;
         };

    }
}

enum RunResult {
    SUCCESS,
    FAILURE
}
