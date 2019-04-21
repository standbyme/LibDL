package vision.datasets;

import LibDL.utils.Pair;
import org.nd4j.linalg.api.ndarray.INDArray;

public class CIFAR100 extends CIFAR10 {

    private static Pair<INDArray, INDArray> read(String root, boolean train) {
        try {
            if (train) return readData(root + "train.bin", 2, 50000);
            else return readData(root + "test.bin", 2, 10000);
        } catch (Exception e) {
            return null;
        }
    }

    public CIFAR100(String root, boolean train) {
        super(read(root, train));
    }
}
