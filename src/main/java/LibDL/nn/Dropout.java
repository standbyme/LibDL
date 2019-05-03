package LibDL.nn;

import LibDL.Tensor.Constant;
import LibDL.Tensor.Tensor;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.ArrayList;

public class Dropout extends Module {

    protected double p;
    protected boolean train;
    protected boolean feature;
    protected boolean alpha;

    protected Dropout(double p, boolean train, boolean feature, boolean alpha) {
        this.p = p;
        this.train = train;
        this.feature = feature;
        this.alpha = alpha;
    }

    public Dropout(double p, boolean train) {
        this(p, train, false, false);
    }

    public Dropout(double p) {
        this(p, true);
    }


    public Tensor forward(Tensor input) {
        return dropout_impl(input, p, train, feature, alpha);
    }

    protected static INDArray make_feature_noise(Tensor input, double p) {
        long[] input_sizes = input.sizes();
        ArrayList<Long> sizes = new ArrayList<>();
        sizes.add(input_sizes[0]);
        sizes.add(input_sizes[1]);
        for (long i = 2; i < input.dim(); i++) {
            sizes.add(new Long(1));
        }
        return (Nd4j.randomBernoulli(p, (long[])
                sizes.stream().mapToLong(i -> i).toArray()
        ));
    }

    protected static Tensor dropout_impl(Tensor tensor, double p,
                                         boolean train,
                                         boolean feature_dropout,
                                         boolean alpha_dropout) {

        if (p == 0 || !train) {
            return tensor;
        }
        if (p == 1) {
            return tensor.mul(0);
        }
        INDArray b;
        INDArray noise = feature_dropout ? make_feature_noise(tensor, 1.0 - p)
                : (Nd4j.randomBernoulli(1.0 - p, tensor.sizes()));
        if (alpha_dropout) {
            INDArray a = Nd4j.ones(tensor.sizes());
            INDArray alpha = Nd4j.create(tensor.sizes());
            alpha.assign(1.7580993408473766);
            a.divi(Transforms.sqrt(alpha.mul(alpha).muli(p).addi(1)
                    .muli(Nd4j.ones(tensor.sizes()).subi(p))));
            b = noise.add(-1).muli(alpha.mul(a))
                    .add(alpha.mul(a).muli(p));
            noise.muli(a);
            return tensor.mul(new Constant(noise)).add(new Constant(b));
        } else {
            noise.divi(1 - p);
            return tensor.mul(new Constant(noise));
        }
    }
}
