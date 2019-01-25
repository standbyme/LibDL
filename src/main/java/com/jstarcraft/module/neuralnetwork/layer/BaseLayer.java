package com.jstarcraft.module.neuralnetwork.layer;

import com.jstarcraft.module.neuralnetwork.activation.ActivationFunction;

public interface BaseLayer {

    ActivationFunction getActivationFunction();

    void setActivationFunction(ActivationFunction activationFunction);

    void setName(String name);

    String getName();

    void doForward();

    void doBackward();
}
