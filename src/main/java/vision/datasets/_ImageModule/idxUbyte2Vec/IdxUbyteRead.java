package vision.datasets._ImageModule.idxUbyte2Vec;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.BufferedInputStream;
import java.io.FileInputStream;
import java.io.InputStream;

/**
 * 内含从MNIST数据集文件(IdxUbyte类型)中读取数据到INDArray的方法<br>
 * 关于该类文件数据的格式 可以查看http://yann.lecun.com/exdb/mnist/
 */
public class IdxUbyteRead {

    /**
     * 从文件中读取数据
     *
     * @param filePath 文件的路径
     * @return
     */
    public static INDArray fromFile(String filePath) {
        return fromFile(filePath, 2048);
    }

    /**
     * 从文件中读取数据 并指定了读取文件时的缓冲数组长度
     *
     * @param filePath      文件的路径
     * @param fileReadBatch 读取文件时的缓冲数组长度
     * @return
     */
    public static INDArray fromFile(String filePath, int fileReadBatch) {
        try {
            BufferedInputStream inputStream = new BufferedInputStream(new FileInputStream(filePath));
            int ndim = readInt(inputStream) - 2048, size = 1;
            int[] shape = (ndim == 1) ? (new int[]{0, 1}) : (new int[ndim]);
            for (int i = 0; i < ndim; i++) shape[i] = readInt(inputStream);
            byte[] data = new byte[fileReadBatch];
            INDArray res = Nd4j.create(shape);
            long rank = 0;
            int mask = 0b11111111;
            while ((size = inputStream.read(data)) != -1) {
                for (int i = 0; i < size; i++) {
                    res.putScalar(rank++, mask & data[i]);
                    if (rank >= res.data().length()) break;
                }
            }
            inputStream.close();
            return res;
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    /**
     * 从InputStream中读取4字节的数据并转换成int值返回
     *
     * @param inputStream
     * @return
     * @throws Exception
     */
    private static int readInt(InputStream inputStream) throws Exception {
        byte[] intReader = new byte[4];
        int size = inputStream.read(intReader);
        return byteConvertToInt(intReader, 0);
    }

    public static int[] byteConvertToInts(byte[] bytes) throws Exception {
        int[] buff = new int[bytes.length / 4];
        for (int i = 0; i < buff.length; i++) {
            buff[i] = byteConvertToInt(bytes, i * 4);
        }
        return buff;
    }

    private static int byteConvertToInt(byte[] bytes, int start) throws Exception {
        int res = 0;
//        if (bytes.length != 4) throw new Exception("File format error.");
        for (int i = start; i < start + 4; i++) {
            int toAdd = (int) bytes[i] + (bytes[i] >= 0 ? 0 : 256);
            res = (res << 8) + toAdd;
        }
        return res;
    }
}
