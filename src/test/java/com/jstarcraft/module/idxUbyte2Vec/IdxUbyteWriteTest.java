package com.jstarcraft.module.idxUbyte2Vec;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class IdxUbyteWriteTest {

    public void toFileTest() {
        INDArray indArray = Nd4j.create(new double[]{1, 2, 3, 4, 5, 6}, new int[]{2, 3});
        String path = "F:/Programs/moduleimage/out/out.idx2-ubyte";

        IdxUbyteWrite.toFile(indArray, path);

        System.out.println(indArray);
        System.out.println(IdxUbyteRead.fromFile("F:/Programs/moduleimage/out/out.idx2-ubyte"));
    }
}
