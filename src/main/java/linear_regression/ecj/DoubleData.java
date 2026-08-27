package linear_regression.ecj;
import ec.gp.GPData;

public class DoubleData extends GPData {
    public double x;

    public void copyTo(final GPData gpd) {
        ((DoubleData)gpd).x = x;
    }
}
