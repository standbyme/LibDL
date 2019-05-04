package vision.datasets;

import LibDL.utils.Pair;
import LibDL.utils.data.Dataset;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.BufferedInputStream;
import java.io.FileInputStream;
import java.io.InputStream;

public class CIFAR10 extends Dataset {


    private static String[] data_paths = {
            "data_batch_1.bin",
            "data_batch_2.bin",
            "data_batch_3.bin",
            "data_batch_4.bin",
            "data_batch_5.bin"
    };
    private static String test_path = "train_batch.bin";

    protected static Pair<INDArray, INDArray> readData(String path, int labelsz, int totalSize) throws Exception {
        InputStream inStream = null;
        BufferedInputStream bis = null;
//        int totalSize = 10000;
        int pixel = 1024;
        int channel = 3;
        INDArray data = Nd4j.create(totalSize, channel * pixel);
        INDArray target = Nd4j.create(totalSize * labelsz);
        byte[] label = new byte[labelsz];
        byte[] allLabel = new byte[totalSize];
        byte[][] data_ch = new byte[channel][pixel];
        try {
            inStream = new FileInputStream(path);
            bis = new BufferedInputStream(inStream);
            for (int i = 0; i < totalSize; i++) {
                bis.read(label);
                for (int ch = 0; ch < channel; ch++) {
                    bis.read(data_ch[ch]);
                    for (int j = 0; j < data_ch[ch].length; j++) {
                        data.put(i, ch * pixel + j, data_ch[ch][j]);
                    }
                }
                for (int j = 0; j < labelsz; j++) {
                    allLabel[i * labelsz + j] = label[j];
                }
            }

            for (int i = 0; i < allLabel.length; i++) {
                target.putScalar(i, allLabel[i]);
            }
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            if (inStream != null) inStream.close();
            if (bis != null) bis.close();
        }
        return new Pair<>(data.reshape(totalSize, channel, pixel), target.reshape(totalSize, labelsz));
    }

    private static Pair<INDArray, INDArray> read(String root, boolean train) {
        Pair<INDArray, INDArray> ret;
        try {
            if (train) {
                ret = readData(root + data_paths[0], 1, 10000);
                for (int i = 1; i < data_paths.length; i++) {
                    Pair<INDArray, INDArray> now = readData(root + data_paths[i], 1, 10000);
                    ret.first = Nd4j.concat(0, ret.first, now.first);
                    ret.second = Nd4j.concat(0, ret.second, now.second);
                }
            } else {
                ret = readData(root + test_path, 1, 10000);
            }

            return ret;
        } catch (Exception e) {
            return null;
        }

    }

    protected CIFAR10(Pair<INDArray, INDArray> pair) {
        super(pair);
    }

    public CIFAR10(String root, boolean train) {
        this(read(root, train));
    }

}
