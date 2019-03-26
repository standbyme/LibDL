package vision.datasets._ImageModule.idxUbyte2Vec;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.BufferedOutputStream;
import java.io.FileOutputStream;
import java.io.IOException;
/**
 * 内含将INDArray中的数据写到IdxUbyte类型文件中的方法<br>
 *     关于该类文件数据的格式 可以查看http://yann.lecun.com/exdb/mnist/
 */
public class IdxUbyteWrite {
    /**
     * 将INDArray中的数据写到IdxUbyte类型文件中
     * @param indArray
     * @param filePath 待写入的文件路径
     */
    public static void toFile(INDArray indArray, String filePath) {
        toFile(indArray, filePath, 2048);
    }

    /**
     * 将INDArray中的数据写到IdxUbyte类型文件中 并指定了每次写入的缓存数组长度
     * @param indArray
     * @param filePath 待写入的文件路径
     * @param fileWriteBatch 写入文件时的缓存数组长度
     */
    public static void toFile(INDArray indArray, String filePath, int fileWriteBatch) {
        try {
            BufferedOutputStream outputStream = new BufferedOutputStream(new FileOutputStream(filePath));
            long[] shape = indArray.shape();
            writeInt(outputStream, shape.length + 2048);
            for (long s : shape) writeInt(outputStream, (int) s);
            DataBuffer data = indArray.data();
            long length = data.length();
            int mask = 0b11111111;

            byte[] buffer = new byte[fileWriteBatch];
            for (int i = 0; i < length; i++) {
                buffer[i % fileWriteBatch] = (byte) (data.getInt(i) & mask);
                if ((i + 1) % fileWriteBatch == 0 || i + 1 == length) {
                    outputStream.write(buffer);
                }
            }
            outputStream.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * 将一个int数据写到BufferedOutputStream中
     * @param outputStream
     * @param out
     * @throws IOException
     */
    private static void writeInt(BufferedOutputStream outputStream, int out) throws IOException {
        int mask = 0b11111111;
        byte[] bytes = new byte[4];
        for (int i = 3; i >= 0; i--) {
            bytes[i] = (byte) (out & mask);
            out = out >> 8;
        }
        outputStream.write(bytes);
    }
}
