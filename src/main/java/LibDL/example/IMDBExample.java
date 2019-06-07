package LibDL.example;

import LibDL.Tensor.Tensor;
import LibDL.nn.*;
import org.apache.commons.lang3.StringUtils;

import java.io.File;
import java.nio.file.Files;
import java.util.*;

class SentimentNet extends Module {

    public Module embedding, rnn, fc;

    public int num_layers, num_hiddens, vocab_size, embed_size;

    public SentimentNet(int num_hiddens,
                        int num_layers,
                        int vocab_size,
                        int embed_size) {
        this.num_hiddens = num_hiddens;
        this.num_layers = num_layers;
        this.vocab_size = vocab_size;
        this.embed_size = embed_size;
        embedding = new Embedding(vocab_size, embed_size);
        rnn = new LSTM(embed_size, num_hiddens, num_layers);
        fc = new Dense(num_hiddens, 2);
    }

    public Tensor get_init_state(long batch_size) {
        return Tensor.randn(num_layers, batch_size, num_hiddens);
    }

    @Override
    public Tensor forward(Tensor input) {
        Tensor out = embedding.forward(input);
        out = rnn.forward(out, get_init_state(input.size(2)),
                get_init_state(out.size(2)));
        out = fc.forward(out);
        return Functional.sigmoid(out);
    }
}

public class IMDBExample {

    static Map<String, Integer> word_to_id;
    static Map<Integer, String> id_to_word;
    static Set<String> words;
    static int word_cnt;

    static List<String> read_files(String dir) {
        List<String> list = new ArrayList<>();
        try {
            final File folder = new File(dir);
            for (final File fileEntry : folder.listFiles()) {
                String content = new String(Files.readAllBytes(fileEntry.toPath()));
                list.add(content);
            }
        } catch (Exception e) {
            System.out.println("shit");
        }
        return list;
    }

    /*
    def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s
     */
    static String normalize_string(String s) {
        String s1 = StringUtils.strip(s.toLowerCase());
        s1 = s1.replaceAll("([.!?])", " \1");
        s1 = s1.replaceAll("[^a-zA-Z.!?]+", " ");
        s1 = StringUtils.strip(s1.replaceAll(" +", " "));
        return s1;
    }

    static void add_words(List<String>... lists) {
        for (List<String> list : lists)
            for (String s : list) {
                words.addAll(Arrays.asList(normalize_string(s).split("\n| ")));
            }
    }

    static void init_map() {
        word_to_id = new HashMap<>();
        id_to_word = new HashMap<>();
        word_cnt = 0;
        for (String s : words) {
            word_to_id.put(s, word_cnt);
            id_to_word.put(word_cnt, s);
            word_cnt++;
        }
    }

    List<List<Long>> to_int_and_pad(List<String> list) {
        return null;
    }


    public static void main(String[] args) {
        List<String> train_pos = read_files("resource/aclImdb/train/pos");
        List<String> train_neg = read_files("resource/aclImdb/train/neg");
        List<String> test_pos = read_files("resource/aclImdb/test/pos");
        List<String> test_neg = read_files("resource/aclImdb/neg/neg");

        words = new HashSet<>();

    }
}
