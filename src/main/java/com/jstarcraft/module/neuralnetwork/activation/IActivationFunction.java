package com.jstarcraft.module.neuralnetwork.activation;

import com.jstarcraft.module.math.structure.matrix.Matrix;

public interface IActivationFunction {
     Matrix forward(Matrix input);
     Matrix backward(Matrix in, Matrix epsilon);
}
