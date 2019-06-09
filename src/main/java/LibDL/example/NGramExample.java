package LibDL.example;

import LibDL.Tensor.Tensor;
import LibDL.Tensor.Variable;
import LibDL.nn.Dense;
import LibDL.nn.Embedding;
import LibDL.nn.Functional;
import LibDL.nn.Module;
import LibDL.optim.Optimizer;
import LibDL.optim.SGD;
import org.nd4j.linalg.factory.Nd4j;

import java.util.*;

//This is from https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html

class NGramLanguageModeler extends Module {
    Module embeddings, linear1, linear2;

    NGramLanguageModeler(int vocab_size, int embedding_dim, int context_size) {
        this.embeddings = new Embedding(vocab_size, embedding_dim);
        this.linear1 = new Dense(context_size * embedding_dim, 128);
        this.linear2 = new Dense(128, vocab_size);
    }

    @Override
    public Tensor forward(Tensor input) {
        Tensor embeds = this.embeddings.forward(input);
        embeds = embeds.reshape(1, embeds.numel());
        Tensor out = Functional.relu(this.linear1.forward(embeds));
        out = this.linear2.forward(out);
        return out;
    }
}

public class NGramExample {
    public static int
            CONTEXT_SIZE = 2,
            EMBEDDING_DIM = 10;

    public static void main(String[] args) {
        String[] test_sentence = ("When forty winters shall besiege thy brow,\n" +
                "And dig deep trenches in thy beauty's field,\n" +
                "Thy youth's proud livery so gazed on now,\n" +
                "Will be a totter'd weed of small worth held:\n" +
                "Then being asked, where all thy beauty lies,\n" +
                "Where all the treasure of thy lusty days;\n" +
                "To say, within thine own deep sunken eyes,\n" +
                "Were an all-eating shame, and thriftless praise.\n" +
                "How much more praise deserv'd thy beauty's use,\n" +
                "If thou couldst answer 'This fair child of mine\n" +
                "Shall sum my count, and make my old excuse,'\n" +
                "Proving his beauty by succession thine!\n" +
                "This were to be new made when thou art old,\n" +
                "And see thy blood warm when thou feel'st it cold.").split(" |\n");
        Set<String> vocab = new HashSet<>(Arrays.asList(test_sentence));
        long cnt = 0;
        Map<String, Long> word_to_idx = new HashMap<>();
        Map<Long, String> idx_to_word = new HashMap<>();
        for (String s : vocab) {
            idx_to_word.put(cnt, s);
            word_to_idx.put(s, cnt++);
        }

        NGramLanguageModeler model = new NGramLanguageModeler(vocab.size(), EMBEDDING_DIM, CONTEXT_SIZE);
        Optimizer optimizer = new SGD(model.parameters(), 0.001f);
        int idx = 0;
        for (int epoch = 0; epoch < 500; epoch++) {
            for (int i = 0; i < test_sentence.length - 2; i++) {
                optimizer.zero_grad();
                Tensor context_idxs = new Variable(Nd4j.create(new double[]{
                        word_to_idx.get(test_sentence[i]).doubleValue(),
                        word_to_idx.get(test_sentence[i + 1]).doubleValue()
                }));

                Tensor loss = Functional.cross_entropy(model.forward(context_idxs),
                        new Variable(Nd4j.create(new double[]{
                                word_to_idx.get(test_sentence[i + 2]).doubleValue()
                        }))
                );

                loss.backward();
                optimizer.step();

                idx++;
                if (idx % 500 == 0) {
                    System.out.println(loss);
                }
            }
        }

        for (int i = 0; i < test_sentence.length - 2; i++) {
            optimizer.zero_grad();
            Tensor context_idxs = new Variable(Nd4j.create(new double[]{
                    word_to_idx.get(test_sentence[i]).doubleValue(),
                    word_to_idx.get(test_sentence[i + 1]).doubleValue()
            }));
            Tensor pred = model.forward(context_idxs);
//            System.out.println(Functional.softmax(pred, 1));
//            System.out.println(pred.data.argMax());
//            System.out.println();
            Long val = (long) (pred.data.argMax().getDouble(0));
            System.out.println("Val: " + val + ", Got: " + idx_to_word.get(val) + ", Expected: " + test_sentence[i + 2]);
        }
    }
}
