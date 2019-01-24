package com;

import com.libDL.nn.layer.DenseLayer;
import com.libDL.nn.layer.OutputLayer;
import com.libDL.nn.network.MultiLayerNetwork;
import com.libDL.nn.optimization.SGDOptimizer;
import org.nd4j.linalg.activations.impl.ActivationReLU;
import org.nd4j.linalg.activations.impl.ActivationSoftmax;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.impl.LossMSE;
import org.nd4j.linalg.primitives.Pair;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

public class LinearClassification {
    static ArrayList<Pair<INDArray, INDArray>> trainData = new ArrayList<>();
    static ArrayList<Pair<INDArray, INDArray>> test = new ArrayList<>();

    public static void main(String[] args) throws IOException {
        readData(test, new File("linear/linear_data_eval.csv"));
        readData(trainData, new File("linear/linear_data_train.csv"));
        MultiLayerNetwork mln = new MultiLayerNetwork();
        DenseLayer l1 = new DenseLayer(2, 20, true);
        OutputLayer l2 = new OutputLayer(20, 2, true);
        l1.setActivationFunction(new ActivationReLU());
        l2.setActivationFunction(new ActivationSoftmax());
//        INDArray w1 = Nd4j.randn(new int[]{3, 2}, 2);
//        INDArray w2 = Nd4j.randn(new int[]{1, 3}, 12);
        //l1.setWeight(w1);
        //l2.setWeight(w2);
        mln.addHiddenLayer(l1);
        mln.setOutputLayer(l2);
        SGDOptimizer sgdOptimizer = new SGDOptimizer();
        sgdOptimizer.setLearnRate(0.01);
        sgdOptimizer.setLossFunction(new LossMSE());
        sgdOptimizer.setIterationListener((model, iterCount) -> {
            if (iterCount % 100 == 0) {
                System.out.println("Error:" + sgdOptimizer.getError());
            }
        });
        int i = 0;
        do {
            Pair<INDArray, INDArray> p = trainData.get(i % trainData.size());
            sgdOptimizer.optimize(mln, p.getFirst(), p.getSecond());
            i++;
        } while (i < 10000);

        for (int j = 0; j < test.size(); j++) {
            Pair<INDArray, INDArray> p = test.get(j);
            INDArray predict = mln.predict(p.getFirst());
            System.out.println("predict:" + predict + ";true:" + p.getSecond());
            //System.out.println(new LossMSE().computeScore(predict, p.getSecond(), new ActivationSigmoid(),null, true));
        }
    }

    private static void readData(ArrayList<Pair<INDArray, INDArray>> data, File file) throws IOException {
        BufferedReader reader = new BufferedReader(new FileReader(file));
        String line;
        while ((line = reader.readLine()) != null) {
            String[] lineArr = line.split(",");
            data.add(new Pair<INDArray, INDArray>(Nd4j.create(
                    new double[]{Double.parseDouble(lineArr[1]), Double.parseDouble(lineArr[2])}
                    , new int[]{2, 1}),
                    Nd4j.create(new double[]{Double.parseDouble(lineArr[0]) == 1 ? 1 : 0, Double.parseDouble(lineArr[0]) == 0 ? 1 : 0}, new int[]{2, 1})));

        }

    }
}
