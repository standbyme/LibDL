package LibDL.nn.layer;

import org.apache.commons.math3.util.FastMath;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class DenseLayer extends DefaultLayer {

    public DenseLayer(int nIn, int nOut, boolean hasBias) {
        this.hasBias = hasBias;
        weight = Nd4j.randn(new int[]{nOut, nIn}).muli(FastMath.sqrt(2.0 / (nIn + nOut)));
        if (hasBias) {
            bias = Nd4j.zeros(nOut, 1);
        }
    }

    public DenseLayer() {
    }

    @Override
    public void doForward() {
        preOutput = weight.mmul(input);
        if (hasBias) {
            preOutput.addi(bias);
        }
        output = activationFunction.getActivation(preOutput, false);
    }

    @Override
    public void doBackward() {
        INDArray temp = activationFunction.backprop(preOutput, error).getFirst(); //input (2, 1) temp (3, 1) epsilon (2, 1)
        epsilon = weight.transpose().mmul(temp);
        weightDiff = error.mmul(input.transpose());
        if (hasBias) {
            biasDiff = getError();
        }
    }

    @Override
    public void update() {
        weight.addi(weightDiff.mul(learnRate));
        if (hasBias) {
            bias.addi(biasDiff.mul(learnRate));
        }
    }

    @Override
    public INDArray run(INDArray input) {
        INDArray weightedInput = getWeight().mmul(input).add(getBias()); //W (3, 2)
        INDArray output = this.getActivationFunction().getActivation(weightedInput, false);
        return output;
    }
}
