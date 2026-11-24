package iris_multi_tree_classification.simpleGP.data_set;

public enum IrisClass {
    SETOSA,
    VERSICOLOUR,
    VIRGINICA;

    public static IrisClass from(String s) {
        return switch (s) {
            case "Iris-setosa" -> SETOSA;
            case "Iris-versicolor" -> VERSICOLOUR;
            case "Iris-virginica" -> VIRGINICA;
            default -> throw new IllegalArgumentException("Unable to produce a class corresponding with the string "+s);
        };
    }
}
