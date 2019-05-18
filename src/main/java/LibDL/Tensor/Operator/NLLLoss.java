package LibDL.Tensor.Operator;

import LibDL.Tensor.OperandInfo;
import LibDL.Tensor.OperatorInfo;
import LibDL.Tensor.OperatorTensor;
import LibDL.Tensor.Tensor;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.function.Supplier;

public class NLLLoss extends OperatorTensor {

    private String reduction;

    private NLLLoss(Builder builder) {

        this.reduction = builder.reduction;
        Tensor input = builder.input;
        Tensor target = builder.target;

        long rows = input.data.rows();
        long cols = input.data.columns();
        INDArray indices = Nd4j.linspace(0, cols * (rows - 1), rows).addi(target.data).reshape(1, rows);

        OperandInfo[] operandInfos = {
                new OperandInfo(input, () -> {
                    INDArray result = Nd4j.zeros(rows * cols)
                            .put(indices, Nd4j.valueArrayOf(1, rows, -1)).reshape(rows, cols);
                    if (reduction.equals("none")) {
                        return null;
                    } else if (reduction.equals("sum")) {
                        return result;
                    } else {
                        return result.divi(rows);
                    }
                }),
                new OperandInfo(target, () -> null),
        };

        Supplier<INDArray> forward = () -> {
            // Be careful this get return NOT a view
            INDArray result = input.data.reshape(rows * cols).get(indices).muli(-1);
            if (reduction.equals("none")) {
                return result.reshape(rows);
            } else if (reduction.equals("sum")) {
                return result.sum();
            } else {
                return result.mean();
            }
        };

        OperatorInfo operatorInfo = new OperatorInfo(operandInfos, forward);

        setOperatorInfo(operatorInfo);

    }


    public static class Builder {

        private final Tensor input;
        private final Tensor target;

        private String reduction;

        public Builder(Tensor input, Tensor target) {
            this.input = input;
            this.target = target;
            this.reduction = "mean";
        }

        public Builder reduction(String reduction) {
            this.reduction = reduction;
            return this;
        }

        public NLLLoss build() {
            return new NLLLoss(this);
        }
    }
}
