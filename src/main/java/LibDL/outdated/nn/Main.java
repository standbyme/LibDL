package LibDL.outdated.nn;

import LibDL.outdated.nn.layer.DenseLayer;
import LibDL.outdated.nn.layer.OutputLayer;
import org.apache.commons.math3.util.FastMath;
import org.nd4j.linalg.activations.impl.ActivationReLU;
import org.nd4j.linalg.activations.impl.ActivationSigmoid;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.impl.LossL2;

public class Main {
    /**
     * The input necessary for XOR.
     */
    // z = 2x + y + 5
    public static double[][] XOR_INPUT = {{0.0, 0.0}, {1.0, 0.0},
            {0.0, 1.0}, {1.0, 1.0}};

    /**
     * The ideal data necessary for XOR.
     */
    public static double[][] XOR_IDEAL = {{0.0}, {1.0}, {1.0}, {0.0}};

    public static void main(String[] args) {
        DenseLayer inputLayer = new DenseLayer();
        OutputLayer outputLayer = new OutputLayer();
        inputLayer.setHasBias(false);
        outputLayer.setHasBias(false);

        inputLayer.setActivationFunction(new ActivationReLU());
        inputLayer.setLearnRate(0.01);
        INDArray w1 = Nd4j.randn(new int[]{3, 2}).muli(FastMath.sqrt(2.0 / (2 + 3)));
        INDArray w2 = Nd4j.randn(new int[]{1, 3}).muli(FastMath.sqrt(2.0 / (3 + 1)));


        w1 = Nd4j.randn(new int[]{3, 2}, 2);
        w2 = Nd4j.randn(new int[]{1, 3}, 22);
        inputLayer.setWeight(w1);
        inputLayer.setBias(Nd4j.zeros(3, 1));
        outputLayer.setActivationFunction(new ActivationSigmoid());
        outputLayer.setLearnRate(0.01);
        outputLayer.setWeight(w2);
        outputLayer.setBias(Nd4j.zeros(1, 1));
        int i = 0;
        double er = 1;
        do {
            i++;
            inputLayer.setInput(Nd4j.create(XOR_INPUT[i % 4], new int[]{2, 1}));
            inputLayer.doForward();
            outputLayer.setInput(inputLayer.getOutput());
            //System.out.println("INPUT: " + new Nd4jMatrix(Nd4j.create(XOR_INPUT[i%4])).toString());
            outputLayer.doForward();
            //System.out.println("OUTPUT: "+outputLayer.getOutput());
            INDArray error = Nd4j.create(XOR_IDEAL[i % 4]).sub(outputLayer.getOutput());
            error = new LossL2().computeGradient(Nd4j.create(XOR_IDEAL[i % 4]), outputLayer.getPreOutput(), new ActivationSigmoid(), null);
            error = new ActivationSigmoid().backprop(outputLayer.getPreOutput(), Nd4j.create(XOR_IDEAL[i % 4]).subi(outputLayer.getOutput())).getFirst();
            double score = new LossL2().computeScore(Nd4j.create(XOR_IDEAL[i % 4]), outputLayer.getPreOutput(), new ActivationSigmoid(), null, true);
            //System.out.println("EEEEE:" + error);
            //System.out.println("score:" + score);
            if (i % 1000 == 0) {
                INDArray e = Nd4j.zeros(1, 1);
                for (int j = 0; j < 4; j++) {
                    INDArray o = outputLayer.run(inputLayer.run(Nd4j.create(XOR_INPUT[j % 4], new int[]{2, 1})));
                    e.addi(Nd4j.create(XOR_IDEAL[j % 4]).sub(o).mul(Nd4j.create(XOR_IDEAL[j % 4]).sub(o)).mul(0.5));
                }
                System.out.println("ERROR: " + e.getDouble(0, 0) / 4);
                er = e.getDouble(0, 0) / 4;
                System.out.println("W: " + outputLayer.getWeight());
            }
            outputLayer.setError(error);
            outputLayer.doBackward();
            outputLayer.update();
            inputLayer.setError(outputLayer.getEpsilon());
            inputLayer.doBackward();
            inputLayer.update();
        } while (er > 0.001);

        System.out.println("aaaaaa");
        for (int j = 0; j < 4; j++) {
            INDArray o = outputLayer.run(inputLayer.run(Nd4j.create(XOR_INPUT[j % 4], new int[]{2, 1})));
            System.out.println(o);
        }
        System.out.println(inputLayer.getWeight());
//       MathMatrix m = new Nd4jMatrix(Nd4j.create(new double[]{2.0, 3.0})).multiplyMatrix(new Nd4jMatrix(Nd4j.create(new double[]{2.0, 3.0}).reshape(2,1)),false);
//        System.out.println(m);
//        inputLayer.doBackward();
//        System.out.println(inputLayer.getWeight());
    }
}
