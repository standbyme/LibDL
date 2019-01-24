package com.libDL.nn.layer;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.LinkedList;
import java.util.Queue;

import static com.libDL.nn.layer.Utils.*;

public class PoolingLayer extends DefaultLayer {
    private int outWidth;
    private int outHeight;
    private int stride = 1;
    private int kernelWidth;
    private int kernelHeight;
    private int inputWidth;
    private int inputHeight;
    private PoolMethod poolMethod = new MaxPool();

    public static void main(String[] args) {
        INDArray i = Nd4j.create(new double[]{
                1, 2, 9, 4, 5, 6, 8, 9
        }, new int[]{2, 2, 2});

        System.out.println(i.argMax().getDouble(0));
    }

    @Override
    public void doForward() {

    }

    @Override
    public void doBackward() {

    }

    @Override
    public void update() {

    }

    private INDArray pool(INDArray input, int kernelWidth, int kernelHeight, int stride) {
        int inputWidth = (int) input.shape()[0];
        int inputHeight = (int) input.shape()[1];
        int inputDepth = (int) input.shape()[2];
        int outW = (inputWidth - kernelWidth) / stride + 1;
        int outH = (inputHeight - kernelWidth) / stride + 1;
        INDArray result = Nd4j.zeros(outW, outH, inputDepth);
        for (int i = 0; i < inputDepth; i++) {
            INDArray layer = getZ(input, i);
            INDArray layerResult = Nd4j.zeros(outW, outH);
            for (int j = 0; j < outW; j++) {
                for (int k = 0; k < outH; k++) {
                    INDArray subArr = subArray(layer, j * stride, k * stride, kernelWidth, kernelHeight);
                    double item = poolMethod.pool(subArr);
                    layerResult.putScalar(j, k, item);
                }
            }
            putZ(result, i, layerResult);
        }
        return result;
    }

    @Override
    public INDArray run(INDArray input) {
        return null;
    }

    private interface PoolMethod {

        double pool(INDArray subArr);

        INDArray dePool(INDArray input);

    }

    private class MaxPool implements PoolMethod {
        private Queue<Integer> maxMem = new LinkedList<>();

        @Override
        public double pool(INDArray subArr) {
            int index = subArr.argMax().getInt(0);
            maxMem.add(index);
            return subArr.getDoubleUnsafe(index);
        }

        @Override
        public INDArray dePool(INDArray input) {
            INDArray result = Nd4j.zeros(inputWidth, inputHeight, input.shape()[2]);
            for (int i = 0; i < input.shape()[2]; i++) {
                for (int j = 0; j < input.shape()[0]; j++) {
                    for (int k = 0; k < input.shape()[1]; k++) {
                        double item = input.getDouble(j, k, i);
                        int mem = maxMem.remove();
                        int offsetInner = (int) (mem / kernelWidth) * inputWidth + mem % kernelWidth;
                        int offsetOuter = j * inputWidth * stride + k * stride + inputWidth * inputHeight * i;
                        result.putScalarUnsafe(offsetInner + offsetOuter, item);
                    }
                }
            }
            return result;
        }
    }

    private class MeanPool implements PoolMethod {

        @Override
        public double pool(INDArray subArr) {
            return subArr.meanNumber().doubleValue();
        }

        @Override
        public INDArray dePool(INDArray input) {
            return null;
        }
    }
}
