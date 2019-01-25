package com.jstarcraft.module.neuralnetwork.activation;

import com.jstarcraft.module.math.structure.matrix.Matrix;
import com.jstarcraft.module.math.structure.matrix.Nd4jMatrix;
import com.jstarcraft.module.math.structure.matrix.NdMatrix;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.Sigmoid;
import org.nd4j.linalg.api.ops.impl.transforms.SigmoidDerivative;
import org.nd4j.linalg.factory.Nd4j;

public class SigmoidActivation implements IActivationFunction {

    @Override
    public Matrix forward(Matrix input) {
        if (input instanceof NdMatrix) {
            INDArray inputArray = NdMatrix.class.cast(input).getArray();
            return new NdMatrix(Nd4j.getExecutioner().execAndReturn(new Sigmoid(inputArray)));
        } else{
            throw new UnsupportedOperationException();
        }
    }

    @Override
    public Matrix backward(Matrix in, Matrix epsilon) {
        if (in instanceof NdMatrix) {
            INDArray inputArray = NdMatrix.class.cast(in).getArray();
            INDArray errorArray = Nd4j.getExecutioner().execAndReturn(new SigmoidDerivative(inputArray));
            errorArray.muli(NdMatrix.class.cast(epsilon).getArray());
            return new NdMatrix(errorArray);
        } else{
            throw new UnsupportedOperationException();
        }
    }
}
