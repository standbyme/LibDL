package LibDL.nn;

import LibDL.Tensor.Tensor;
import org.springframework.lang.Nullable;

public class BCELoss extends LossLayer {

    Tensor weight;
    String reduce;

    public BCELoss() {
        this(null, null);
    }

    public BCELoss(@Nullable Tensor weight, @Nullable String reduce) {
        this.weight = weight;
        this.reduce = reduce;
    }

    @Override
    public Tensor forward(Tensor input) {
        throw new UnsupportedOperationException();
    }

    @Override
    public Tensor forward(Tensor... input) {
        assert input.length == 2;
        Tensor x = input[0];
        Tensor y = input[1];
        Tensor ones = Tensor.ones_like(x);
        Tensor out = y.mul(Tensor.log(x)).add(ones.sub(y).mul(Tensor.log(ones.sub(x)))).mul(-1);
        if (weight != null) out = weight.mul(out);
        if (reduce != null) {
            if (reduce.equals("mean")) return out.mean(1);
            else if (reduce.equals("sum")) return out.sum(1);
            else if (reduce.equals("none")) return out;
        }
        return out.mean(1);
    }
}
