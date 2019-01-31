package com.jstarcraft.module.datatran.transform.nd4j;

import com.jstarcraft.module.datatran.RawDataSet;
import com.jstarcraft.module.datatran.transform.BaseTransform;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.Collection;

/**
 * <p>类名CookDSTransform为CookDataSetTransform的简称 顾名思义 该变换是将{@link RawDataSet}变换为nd4j的{@link DataSet}</p>
 * <p>继承自<code>{@link BaseTransform}&lt;{@link RawDataSet}&lt;{@link INDArray},?&gt;, {@link DataSet}&gt;</code> 其中的问号代表{@link INDArray}类型或者{@link Number}类型 {@link Number}类型在该变换过程中被转换成了行数列数均为<code>1</code>的{@link INDArray}</p>
 * <p>将<code>{@link RawDataSet}&lt;{@link INDArray},?&gt;</code>中类型为<code>{@link ArrayList}&lt;{@link INDArray}&gt;</code>的特征集合 通过nd4j提供的{@link Nd4j#vstack(Collection)}方法合并为一个类型为{@link INDArray}的对象 这个对象即为返回的{@link DataSet}对象中的特征集合</p>
 * <p>对标签集合进行同样的操作</p>
 */
public class CookDSTransform extends BaseTransform<RawDataSet<INDArray,?>, DataSet> {
    /**
     * <p>将<code>{@link RawDataSet}&lt;{@link INDArray},?&gt;</code>中类型为<code>{@link ArrayList}&lt;{@link INDArray}&gt;</code>的特征集合 通过nd4j提供的{@link Nd4j#vstack(Collection)}方法合并为一个类型为{@link INDArray}的对象 这个对象即为返回的{@link DataSet}对象中的特征集合</p>
     * <p>对标签集合进行同样的操作</p>
     * @param rawDataSet
     * @return
     */
    @Override
    public DataSet tran2(RawDataSet<INDArray, ?> rawDataSet) {
        ArrayList<INDArray> rawFeatures = rawDataSet.getFeatures();
        ArrayList<?> rawLabels = rawDataSet.getLabels();
        return new DataSet(indArraysVStack(rawFeatures),indArraysVStack(rawLabels));
    }

    /**
     * <p>用{@link Nd4j#vstack(Collection)}的方式合并{@link ArrayList}<code>&lt;?&gt;</code>列表中的对象 得到一个{@link INDArray}对象</p>
     * <p>其中问号代表{@link INDArray}类型或{@link Number}类型</p>
     * @param indArraysIn
     * @return
     */
    private INDArray indArraysVStack(ArrayList<?> indArraysIn) {
        if (indArraysIn.size() <= 0) return null;
        ArrayList<INDArray> indArrays = new ArrayList<>();
        indArraysIn.stream().map(this::toINDArray).forEach(indArrays::add);
        return Nd4j.vstack(indArrays);
    }

    /**
     * 将{@link Number}类型或者{@link INDArray}类型的对象 转换成{@link INDArray}类型的对象
     * @param label
     * @return
     */
    private INDArray toINDArray(Object label) {
        if (label instanceof Number) return Nd4j.ones(1,1).muli((Number) label);
        else return (INDArray)label;
    }
}
