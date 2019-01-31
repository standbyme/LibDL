package com.jstarcraft.module.datatran.transform.rawdataset;

import com.jstarcraft.module.datatran.RawDataSet;
import com.jstarcraft.module.datatran.transform.BaseTransform;

/**
 * <p>{@link RawDataSet}<code>&lt;F,L&gt;</code>与{@link RawDataSet}<code>&lt;F,L&gt;</code>之间的变换 封装为{@link RDSTransform}<code>&lt;F,L&gt;</code></p>
 * <p>RDSTransform是RawDataTransform的简称</p>
 * @param <F> 指定转换前后的{@link RawDataSet}的特征变量的类型
 * @param <L> 指定转换前后的{@link RawDataSet}的标签变量的类型
 */
public abstract class RDSTransform<F,L> extends BaseTransform<RawDataSet<F,L>,RawDataSet<F,L>> {
}
