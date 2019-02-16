package com.jstarcraft.module.idxUbyte2Vec;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.*;

public class IdxUbyteRead {

    public static INDArray fromFile(String filePath) {
        return fromFile(filePath, 2048);
    }

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


    private static int readInt(InputStream inputStream) throws Exception {
        byte[] intReader = new byte[4];
        int size = inputStream.read(intReader), res = 0;
        if (size != 4) throw new Exception("File format error.");
        for (int i = 0; i < 4; i++) res = (res << 8) + intReader[i];
        return res;
    }
}
