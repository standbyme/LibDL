package LibDL.nn;

import LibDL.Tensor.Parameter;
import LibDL.Tensor.Tensor;
import org.apache.commons.lang3.ArrayUtils;
import org.nd4j.linalg.factory.Nd4j;
import org.springframework.lang.Nullable;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Embedding extends Module {
    public int num_embeddings,
            embedding_dim;
    public Integer padding_idx;
    public Float max_norm,
            norm_type;
    public boolean scale_grad_by_freq,
            sparse;

    public Parameter weight;

    public Embedding(int num_embeddings,
                     int embedding_dim,
                     @Nullable Integer padding_idx,
                     @Nullable Float max_norm,//
                     @Nullable Float norm_type,//2.0
                     boolean scale_grad_by_freq,//False
                     boolean sparse//False
    ) {
        this.num_embeddings = num_embeddings;
        this.embedding_dim = embedding_dim;
        this.padding_idx = padding_idx;
        this.max_norm = max_norm;
        this.norm_type = norm_type == null ? 2.0f : norm_type;
        this.scale_grad_by_freq = scale_grad_by_freq;
        this.sparse = sparse;
        this.weight = new Parameter(Nd4j.create(num_embeddings, embedding_dim));
        reset_params();

    }

    public Embedding(int num_embeddings, int embedding_dim) {
        this(num_embeddings, embedding_dim, null, null, null, false, false);
    }

    private void reset_params() {
        WeightInit.normal(weight.data, 0, 1);
        if (padding_idx != null) {
            weight.data.putRow(padding_idx.intValue(), Nd4j.zeros(1, num_embeddings));

        }
    }

    @Override
    public Tensor forward(Tensor index) {
        if (weight.dim() == 1) {
            return weight.index_select(index);
        }
        List<Long> sizes = new ArrayList<>(Arrays.asList(ArrayUtils.toObject(index.sizes())));
        long[] weight_sizes = weight.sizes();
        for (int i = 1; i < weight_sizes.length; i++) {
            sizes.add(weight_sizes[i]);
        }
        return weight.index_select(index.reshape(-1)).reshape(sizes.stream().mapToLong(i -> i).toArray());

    }
}
