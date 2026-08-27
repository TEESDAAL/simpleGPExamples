package iris_multi_tree_classification.simpleGP;


record Features(
    SepalLength sepalLength,
    SepalWidth sepalWidth,
    PetalLength petalLength,
    PetalWidth petalWidth
) {}

record SepalLength(double cm) {}
record SepalWidth(double cm) {}
record PetalLength(double cm) {}
record PetalWidth(double cm) {}

enum Class {
    SETOSA,
    VERSICOLOUR,
    VIRGINICA;
}