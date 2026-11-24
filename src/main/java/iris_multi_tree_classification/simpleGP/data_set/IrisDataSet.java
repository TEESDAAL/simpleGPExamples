package iris_multi_tree_classification.simpleGP.data_set;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVRecord;
import util.DataPoint;
import util.dataset.DataSet;
import util.result.Result;


import java.io.FileReader;
import java.nio.file.Path;
import java.util.List;


public record IrisDataSet(List<DataPoint<Features, IrisClass>> data) implements DataSet<Features, IrisClass> {
    public static Result<IrisDataSet, Throwable> fromCSV(FileReader csvFile) {
         return Result.fromFunction(() -> new IrisDataSet(CSVFormat.DEFAULT.parse(csvFile)
                 .stream()
                 .map(IrisDataSet::fromRecord)
                 .toList()));
    }
    public static Result<IrisDataSet, Throwable> fromCSV(Path csvFile) {
        return Result.fromFunction(() -> new FileReader(csvFile.toFile()))
                .flatMap(IrisDataSet::fromCSV);
    }

    public static DataPoint<Features, IrisClass> fromRecord(CSVRecord record) {
        return DataPoint.of(
                Features.of(
                        Double.parseDouble(record.get(0)),
                        Double.parseDouble(record.get(1)),
                        Double.parseDouble(record.get(2)),
                        Double.parseDouble(record.get(3))
                ), IrisClass.from(record.get(4))
        );
    }
}
