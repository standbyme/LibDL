package vision.datasets;

import LibDL.utils.data.Dataset;
import com.jstarcraft.module.idxUbyte2Vec.IdxUbyteRead;
import org.jetbrains.annotations.NotNull;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class MNIST extends Dataset {
    private static String[] test_name = {
            "t10k-images-idx3-ubyte",
            "t10k-labels-idx1-ubyte",
    };
    private static String[] train_name = {
            "train-images-idx3-ubyte",
            "train-labels-idx1-ubyte"
    };
    public INDArray train_data, train_label, test_data, test_label;

    public MNIST(String root, boolean train) {
        if (train) {
            train_data = IdxUbyteRead.fromFile(root + train_name[0], 1000);
            train_label = IdxUbyteRead.fromFile(root + train_name[1], 1000);
        } else {
            test_data = IdxUbyteRead.fromFile(root + test_name[0], 1000);
            test_label = IdxUbyteRead.fromFile(root + test_name[1], 1000);
        }
    }
}
