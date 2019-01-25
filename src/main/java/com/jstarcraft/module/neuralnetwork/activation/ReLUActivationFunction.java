package com.jstarcraft.module.neuralnetwork.activation;

import com.jstarcraft.module.math.structure.MathCalculator;
import com.jstarcraft.module.math.structure.matrix.MathMatrix;
import com.jstarcraft.module.math.structure.matrix.Nd4jMatrix;
import com.jstarcraft.module.math.structure.vector.MathVector;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.RectifedLinear;
import org.nd4j.linalg.api.ops.impl.transforms.Sigmoid;
import org.nd4j.linalg.api.ops.impl.transforms.SigmoidDerivative;
import org.nd4j.linalg.api.ops.impl.transforms.Step;
import org.nd4j.linalg.factory.Nd4j;

/**
 * f(x) = max(0, x)
 */
public class ReLUActivationFunction implements ActivationFunction {

	@Override
	public MathMatrix forward(MathMatrix input){
		if (input instanceof Nd4jMatrix) {
			INDArray inputArray = Nd4jMatrix.class.cast(input).getArray();
			return new Nd4jMatrix(Nd4j.getExecutioner().execAndReturn(new RectifedLinear(inputArray)));
		} else{
			throw new UnsupportedOperationException();
		}
	}

	@Override
	public MathMatrix backward(MathMatrix input, MathMatrix epsilon){
		if (input instanceof Nd4jMatrix) {
			INDArray inputArray = Nd4jMatrix.class.cast(input).getArray();
			INDArray errorArray = Nd4j.getExecutioner().execAndReturn(new Step(inputArray)); //(1, 1)
			errorArray.muli(Nd4jMatrix.class.cast(epsilon).getArray());
			return new Nd4jMatrix(errorArray);
		} else{
			throw new UnsupportedOperationException();
		}
	}


	@Override
	public void forward(MathMatrix input, MathMatrix output) {
		output.mapValues((row, column, value, message) -> {
			value = input.getValue(row, column);
			value = value < 0F ? 0F : value;
			return value;
		}, null, MathCalculator.PARALLEL);
	}

	@Override
	public void forward(MathVector input, MathVector output) {
		output.mapValues((index, value, message) -> {
			value = input.getValue(index);
			value = value < 0F ? 0F : value;
			return value;
		}, null, MathCalculator.SERIAL);
	}

	@Override
	public void backward(MathMatrix input, MathMatrix error, MathMatrix output) {
		output.mapValues((row, column, value, message) -> {
			value = input.getValue(row, column);
			value = (value > 0F ? 1F : 0F);
			value *= error.getValue(row, column);
			return value;
		}, null, MathCalculator.PARALLEL);
	}

	@Override
	public void backward(MathVector input, MathVector error, MathVector output) {
		output.mapValues((index, value, message) -> {
			value = input.getValue(index);
			value = (value > 0F ? 1F : 0F);
			value *= error.getValue(index);
			return value;
		}, null, MathCalculator.SERIAL);
	}

	@Override
	public boolean equals(Object object) {
		if (this == object) {
			return true;
		}
		if (object == null) {
			return false;
		}
		if (getClass() != object.getClass()) {
			return false;
		} else {
			return true;
		}
	}

	@Override
	public int hashCode() {
		return getClass().hashCode();
	}

	@Override
	public String toString() {
		return "ReLUActivationFunction()";
	}

}
