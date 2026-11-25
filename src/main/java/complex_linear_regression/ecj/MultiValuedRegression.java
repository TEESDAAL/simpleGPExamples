package complex_linear_regression.ecj;

import ec.util.*;
import ec.*;
import ec.gp.*;
import ec.gp.koza.*;
import ec.simple.*;

import java.util.function.DoubleBinaryOperator;
import java.util.function.Supplier;

public class MultiValuedRegression extends GPProblem implements SimpleProblemForm {
    public static final String P_DATA = "data";

    public double currentX;
    public double currentY;

    public void setup(final EvolutionState state,
                      final Parameter base) {
        // very important, remember this
        super.setup(state, base);

        // verify our input is the right class (or subclasses from it)
        if (!(input instanceof DoubleData))
            state.output.fatal("GPData class must subclass from " + DoubleData.class,
                base.push(P_DATA), null);
    }

    public void evaluate(final EvolutionState state,
                         final Individual ind,
                         final int subpopulation,
                         final int threadnum) {
       DoubleData input = (DoubleData) (this.input);

        var res = computeScore(
                state.random[threadnum],
                (x, y) -> {
                    // fantastic design here folks...
                    this.currentX = x;
                    this.currentY = y;
                    ((GPIndividual) ind).trees[0].child.eval(
                            state,
                            threadnum,
                            input,
                            stack,
                            ((GPIndividual) ind),
                            this
                    );
                    return input.x;
                }
        );

        // the fitness better be KozaFitness!
        KozaFitness f = ((KozaFitness) ind.fitness);
        f.setStandardizedFitness(state, res.score());
        f.hits = res.hits();
        ind.evaluated = true;
    }

    public static ECJFitnessInfo computeScore(
            MersenneTwisterFast random,
            DoubleBinaryOperator model
    ) {
        int hits = 0;
        double sum = 0.0;
        double expectedResult;
        double result;
        for (int y = 0; y < 100; y++) {
            double currentX = random.nextDouble() * 2 * Math.PI - Math.PI;
            double currentY = random.nextDouble() * 2 * Math.PI - Math.PI;
            expectedResult = Math.cos(currentY) * Math.exp(Math.sin(currentX));
            result = Math.abs(expectedResult - model.applyAsDouble(currentX, currentY));
            if (result <= 0.01) hits++;
            sum += result;
        }
        return new ECJFitnessInfo(sum, hits);
    }
}
