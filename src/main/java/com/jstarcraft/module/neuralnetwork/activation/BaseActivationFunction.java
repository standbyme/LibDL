package com.jstarcraft.module.neuralnetwork.activation;

import com.jstarcraft.module.math.structure.matrix.MathMatrix;
import com.jstarcraft.module.math.structure.vector.MathVector;

public abstract class BaseActivationFunction implements ActivationFunction {
    @Override
    public void forward(MathMatrix input, MathMatrix output) {

    }

    @Override
    public void forward(MathVector input, MathVector output) {

    }

    @Override
    public void backward(MathMatrix input, MathMatrix error, MathMatrix output) {

    }

    @Override
    public void backward(MathVector input, MathVector error, MathVector output) {

    }
}
