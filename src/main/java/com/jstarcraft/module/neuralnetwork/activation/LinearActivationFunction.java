package com.jstarcraft.module.neuralnetwork.activation;

import com.jstarcraft.module.math.structure.matrix.MathMatrix;

public class LinearActivationFunction extends BaseActivationFunction{
    @Override
    public MathMatrix backward(MathMatrix in, MathMatrix epsilon){
        return in.dotProduct(epsilon,false);
    }

    @Override
    public MathMatrix forward(MathMatrix input) {
        return input;
    }
}
