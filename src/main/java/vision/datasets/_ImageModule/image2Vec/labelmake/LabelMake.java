package vision.datasets._ImageModule.image2Vec.labelmake;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;
import java.util.stream.Collectors;

/**
 * 包含和标签生成相关的方法
 */
public class LabelMake {
    /**
     * 传入一组特征及对应的从特征到标签的转换关系 得到指示这一组标签的INDArray
     *
     * @param features
     * @param function 将特征对象转换为与其对应的 指示了一个标签值的INDArray类型的对象
     * @param shape    指定返回的INDArray的shape
     * @param <F>
     * @return
     */
    public static <F> INDArray labelMake(List<F> features, Function<F, INDArray> function, int... shape) {
        ArrayList<INDArray> indArrays = features.stream().map(function).collect(Collectors.toCollection(ArrayList::new));
        return Nd4j.create(indArrays, shape);
    }

    /**
     * 从{@link File}对象获得它所在文件夹的名字 作为该对象的标签 该方法是由特征对象生成标签对象的方法之一
     *
     * @param feature
     * @return
     */
    public static String parentPathLM(File feature) {
        return feature.getParentFile().getName();
    }
}
