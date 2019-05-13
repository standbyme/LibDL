package LibDL.example;

import LibDL.Tensor.Tensor;
import LibDL.Tensor.Variable;
import LibDL.nn.Dense;
import LibDL.nn.Functional;
import LibDL.nn.Module;
import LibDL.nn.RNNAuto;
import LibDL.optim.RMSProp;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.indexaccum.IMax;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;

public class RNNExample {

    private static final char[] LEARNSTRING =
            "*This example trains a RNN with LibDL, which recites this sentence.".toCharArray();
    private static final List<Character> LEARNSTRING_CHARS_LIST = new ArrayList<>();

    static class Model extends Module {
        private RNNAuto rnn;
        private Dense output;

        Model(int inputSize, int hiddenSize) {
            rnn = new RNNAuto(inputSize, hiddenSize, 1);
            output = new Dense(hiddenSize, hiddenSize);
        }

        @Override
        public Tensor forward(Tensor input) {
            Tensor x;
            x = rnn.forward(input);
            x = output.forward(x);
            return x;
        }
    }

    public static void main(String[] args) {

        LinkedHashSet<Character> LEARNSTRING_CHARS = new LinkedHashSet<>();
        for (char c : LEARNSTRING)
            LEARNSTRING_CHARS.add(c);
        LEARNSTRING_CHARS_LIST.addAll(LEARNSTRING_CHARS);
        int seqlen = LEARNSTRING.length;
        int features = LEARNSTRING_CHARS_LIST.size();

        INDArray input = Nd4j.zeros(seqlen, 1, LEARNSTRING_CHARS_LIST.size());
        INDArray label = Nd4j.zeros(seqlen);
        // loop through our sample-sentence
        int samplePos = 0;
        for (char currentChar : LEARNSTRING) {
            // small hack: when currentChar is the last, take the first char as
            // nextChar - not really required. Added to this hack by adding a starter first character.
            char nextChar = LEARNSTRING[(samplePos + 1) % (LEARNSTRING.length)];
            // input neuron for current-char is 1 at "samplePos"
            input.putScalar(new int[] { samplePos, 0, LEARNSTRING_CHARS_LIST.indexOf(currentChar) }, 1);
            // output neuron for next-char is 1 at "samplePos"
            label.putScalar(new int[] { samplePos }, LEARNSTRING_CHARS_LIST.indexOf(nextChar));
            samplePos++;
        }
        System.out.println("input: " + tensor2str(input));

        Variable inputTensor = new Variable(input);
        Variable labelTensor = new Variable(label);

        Model model = new Model(features, features);
        System.out.println(model);
        RMSProp optim = new RMSProp(model.parameters(), 0.001f, 0.99f, 5e-8);

        for (int epoch = 0; epoch < 500; epoch++) {
            optim.zero_grad();
            Tensor result = model.forward(inputTensor);
            System.out.println(epoch + ": " + tensor2str(result.data));
//            System.out.println(Arrays.toString(result.data.shape()));
//            System.out.println(Arrays.toString(labelTensor.data.shape()));

            Tensor loss = Functional.cross_entropy(result.reshape(seqlen, features), labelTensor);
            loss.backward();
            optim.step();
        }

    }

    private static String tensor2str(INDArray input) {
        StringBuilder res = new StringBuilder();
        INDArray characterIdx = Nd4j.getExecutioner().exec(new IMax(input), input.rank()-1);
        for(int i = 0; i < characterIdx.length(); i++) {
            res.append(LEARNSTRING_CHARS_LIST.get(characterIdx.getInt((i))));
        }
        return res.toString();
    }
}
