/*-
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package com.jstarcraft.module.neuralnetwork.layer;

import java.util.Map;

import com.jstarcraft.core.utility.KeyValue;
import com.jstarcraft.module.math.structure.matrix.MathMatrix;
import com.jstarcraft.module.neuralnetwork.MatrixFactory;
import com.jstarcraft.module.neuralnetwork.Model;
import com.jstarcraft.module.neuralnetwork.activation.ActivationFunction;

/**
 * Interface for a layer of a neural network. This has an activation function,
 * an input and output size, weights, and a bias
 *
 * @author Adam Gibson
 */
public interface Layer extends Model {

	@Deprecated
	enum Mode {

		TRAIN,

		TEST;

	}

	/**
	 * 根据指定的样本分配缓存(每次epoch调用)
	 * 
	 * @param factory
	 * @param samples
	 */
	void doCache(MatrixFactory factory, KeyValue<MathMatrix, MathMatrix> samples);

	/**
	 * A representation of the vertices that are inputs to this vertex (inputs duing
	 * forward pass)<br>
	 * Specifically, if inputVertices[X].getVertexIndex() = Y, and
	 * inputVertices[X].getVertexEdgeNumber() = Z then the Zth output connection
	 * (see {@link #getNumberOfOutputs()} of vertex Y is the Xth input to this
	 * vertex
	 */
	KeyValue<MathMatrix, MathMatrix> getInputKeyValue();

	KeyValue<MathMatrix, MathMatrix> getMiddleKeyValue();

	/**
	 * A representation of the vertices that this vertex is connected to (outputs
	 * duing forward pass) Specifically, if outputVertices[X].getVertexIndex() = Y,
	 * and outputVertices[X].getVertexEdgeNumber() = Z then the Xth output of this
	 * vertex is connected to the Zth input of vertex Y
	 */
	KeyValue<MathMatrix, MathMatrix> getOutputKeyValue();

	float calculateL1Norm();

	float calculateL2Norm();

	void regularize();

	Map<String, MathMatrix> getParameters();

	Map<String, MathMatrix> getGradients();

	void setMode(Mode mode);

	Mode getMode();

	ActivationFunction getFunction();

}
