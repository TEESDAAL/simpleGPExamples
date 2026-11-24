package iris_multi_tree_classification.simpleGP.data_set;

import iris_multi_tree_classification.simpleGP.data_set.units.PetalLength;
import iris_multi_tree_classification.simpleGP.data_set.units.PetalWidth;
import iris_multi_tree_classification.simpleGP.data_set.units.SepalLength;
import iris_multi_tree_classification.simpleGP.data_set.units.SepalWidth;

public record Features(
    SepalLength sepalLength,
    SepalWidth sepalWidth,
    PetalLength petalLength,
    PetalWidth petalWidth
) {
    public static Features of(double sepalLength, double sepalWidth, double petalLength, double petalWidth) {
        return new Features(
                new SepalLength(sepalLength),
                new SepalWidth(sepalWidth),
                new PetalLength(petalLength),
                new PetalWidth(petalWidth)
        );
    }
}
