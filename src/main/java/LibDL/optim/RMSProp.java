package LibDL.optim;

import LibDL.Tensor.Variable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.Arrays;

public class RMSProp extends Optimizer {

    private INDArray[] Sdparams;
    private float beta, alpha;
    private double eps;

    public RMSProp(Parameters parameters, float lr, float alpha, double eps) {
        super(parameters);
        this.alpha = lr;
        this.beta = alpha;
        this.eps = eps;
    }


    @Override
    public void step() {
        if(params == null) {
            cacheParams();
            this.Sdparams = Arrays.stream(params)
                    .map(constant -> Nd4j.zerosLike(constant.value))
                    .toArray(INDArray[]::new);
        }

        for (int i = 0; i < params.length; i++) {
            Variable param = params[i];
            Sdparams[i].muli(beta).addi
                    (param.dout.mul(param.dout).muli(1.0 - beta));
            param.value.subi(param.dout.mul(alpha).divi(Transforms.sqrt(Sdparams[i]).add(eps)));
        }
    }
}