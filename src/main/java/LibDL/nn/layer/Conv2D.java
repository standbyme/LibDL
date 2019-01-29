package LibDL.nn.layer;

import org.nd4j.linalg.activations.impl.ActivationIdentity;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import static LibDL.nn.layer.Utils.*;

public class Conv2D extends DefaultLayer {
    private int outHeight;
    private int outWidth;
    private int stride = 1;
    private INDArray rotatedKernel;

    public static void main(String[] args) {
        Conv2D convLayer = new Conv2D();
        convLayer.setOutHeight(2);
        convLayer.setOutWidth(2);
        convLayer.setWeight(Nd4j.ones(5, 5, 1));
        convLayer.setActivationFunction(new ActivationIdentity());
        INDArray input = Nd4j.ones(28, 28, 1);
        convLayer.setInput(input);
        convLayer.doForward();
        System.out.println(convLayer.getOutput());
    }

    @Override
    public void doForward() {
        preOutput = conv2D(input, weight);
        output = activationFunction.getActivation(preOutput, true);
    }

    @Override
    public void doBackward() {
        INDArray temp = activationFunction.backprop(preOutput, error).getFirst();
        rotatedKernel = weight.dup();
        for (int i = 0; i < rotatedKernel.shape()[2]; i++) {
            INDArray t = getZ(rotatedKernel, i);
            t = Nd4j.reverse(t);
            putZ(rotatedKernel, i, t);
        }
        INDArray paddedRotatedKernel = Nd4j.pad(rotatedKernel, new int[]{1, 1, 0}, Nd4j.PadMode.CONSTANT);
        epsilon = conv2D(paddedRotatedKernel, temp);
        weightDiff = conv2D(input, error);
        if (hasBias) {
            biasDiff = error;
        }
    }

    @Override
    public void update() {
        weight.addi(weightDiff);
        if (hasBias) {
            bias.addi(biasDiff);
        }
    }

    @Override
    public INDArray run(INDArray input) {
        INDArray preOutput = conv2D(input, weight);
        return activationFunction.getActivation(preOutput, true);
    }

    private INDArray conv2D(INDArray input, INDArray kernel) {
        INDArray result = Nd4j.zeros(outHeight, outWidth, kernel.shape()[2]);
        for (int z = 0; z < kernel.shape()[2]; z++) {
            INDArray k = getZ(kernel, z);
            INDArray channelSum = Nd4j.zeros(outHeight, outWidth);
            for (int c = 0; c < input.shape()[2]; c++) {
                INDArray channel = getZ(input, c);
                INDArray convChannel = Nd4j.zeros(outHeight, outWidth);
                for (int i = 0; i < outHeight; i++) {
                    for (int j = 0; j < outWidth; j++) {
                        INDArray subArr = subArray(channel, i * stride, j * stride, k.shape()[0], k.shape()[1]);
                        double item = subArr
                                .muli(k)
                                .sumNumber().doubleValue();
                        convChannel.putScalar(i, j, item);
                    }
                }
                channelSum.addi(convChannel);
            }
            putZ(result, z, channelSum);
        }
        return result;
    }

    public void setOutHeight(int outHeight) {
        this.outHeight = outHeight;
    }

    public void setOutWidth(int outWidth) {
        this.outWidth = outWidth;
    }
}
