package com.jstarcraft.module.datatran.translator;

import com.jstarcraft.module.datatran.entity.FLPair;
import com.jstarcraft.module.datatran.function.labelmake.LabelMake;
import com.jstarcraft.module.datatran.translator.indexmake.IndexMake;
import com.jstarcraft.module.datatran.translator.select.Select;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;


import java.util.*;
import java.util.function.Consumer;
import java.util.function.Function;

/**
 * 比较核心的类 其中的{@link Translator#data}为一个数据集 并封装了一些{@link Translator}之间的变换方法
 * @param <F> 数据集的特征对象类型
 * @param <L> 数据集的标签对象类型
 */
public class Translator<F,L> {
    /**
     * 数据集
     */
    private List<FLPair<F,L>> data;

    /**
     * 构造函数
     * @param data
     */
    public Translator(List<FLPair<F, L>> data) {
        setData(data);
    }

    /**
     * 静态方法 由一个{@link List}&lt;{@link FLPair}&lt;F,L&gt;&gt;对象 初始化生成一个{@link Translator}&lt;F,L&gt;对象
     * @param data
     * @param <F>
     * @param <L>
     * @return
     */
    public static <F,L> Translator<F,L> init(List<FLPair<F,L>> data) {
        return new Translator<>(data);
    }

    /**
     * 静态方法 传入一组特征对象 以及由特征对象生成标签对象的{@link Function}&lt;F,L&gt;方法
     *  初始化生成一个{@link Translator}&lt;F,L&gt;对象
     * @param features
     * @param function
     * @param <F>
     * @param <L>
     * @return
     */
    public static <F,L> Translator<F,L> init(List<F> features, Function<F,L> function) {
        return Translator.init(LabelMake.makeAll(features,function));
    }

    /**
     * 静态方法 将一个{@link Translator}&lt;{@link INDArray},{@link INDArray}&gt;对象转换为ND4J的{@link DataSet}
     *  将{@link Translator#data}中的一组特征对象 和{@link Translator#data}中的一组标签对象(均为{@link INDArray}格式)
     *   通过{@link Nd4j#vstack(Collection)}方法合并为两个{@link INDArray}对象 以此调用{@link DataSet#DataSet(INDArray, INDArray)}方法
     *    生成对应的{@link DataSet}
     * @param translator
     * @return
     */
    public static DataSet asDataSet(Translator<INDArray,INDArray> translator) {
        INDArray feature = Nd4j.vstack(translator.stream().getFeatureList());
        INDArray label = Nd4j.vstack(translator.stream().getLabelList());
        return new DataSet(feature,label);
    }

    /**
     * 调用了{@link Translator#data}的{@link List#forEach(Consumer)}方法
     * @param consumer
     */
    public void forEach(Consumer<FLPair<F,L>> consumer) {
        data.forEach(consumer);
    }

    /**
     * 对数据集中的每一个特征变量执行传入的consumer操作
     * @param consumer
     */
    public void forEachF(Consumer<F> consumer) {
        data.forEach(p->consumer.accept(p.getFeature()));
    }

    /**
     * 对数据集中的每一个标签变量执行传入的consumer操作
     * @param consumer
     */
    public void forEachL(Consumer<L> consumer) {
        data.forEach(p->consumer.accept(p.getLabel()));
    }

    /**
     * 生成该对象对应的{@link TranslatorStream} 以此执行函数式编程相关的操作
     * @return
     */
    public TranslatorStream<F,L> stream() {
        return new TranslatorStream<>(data.stream());
    }

    /**
     * 调用了{@link Translator#shuffle(Random)}
     * @return
     */
    public Translator<F, L> shuffle() {
        return shuffle(new Random());
    }

    /**
     * 对{@link Translator#data}调用了{@link Select#shuffle(List, Random)}
     * @param random
     * @return
     */
    public Translator<F, L> shuffle(Random random) {
        data = Select.shuffle(data,random);
        return this;
    }

    /**
     * 调用了{@link Translator#sample(int, Random)}
     * @param count
     * @return
     */
    public Translator<F, L> sample(int count) {
        return sample(count, new Random());
    }

    /**
     * 对{@link Translator#data}调用了{@link Select#sample(List, int, Random)}
     * @param count
     * @param random
     * @return
     */
    public Translator<F, L> sample(int count, Random random) {
        data = Select.sample(data,count,random);
        return this;
    }

    /**
     * 调用了{@link Translator#balanceLabel(Random)}
     * @return
     */
    public Translator<F, L> balanceLabel() {
        return balanceLabel(new Random());
    }

    /**
     * 对{@link Translator#data}调用了{@link Select#balanceLabel(List, Random)}
     * @param random
     * @return
     */
    public Translator<F, L> balanceLabel(Random random) {
        data = Select.balanceLabel(data, random);
        return this;
    }

    /**
     * 调用了{@link IndexMake#indexMake(List)}生成新的数据集 返回由新数据集初始化而成的{@link Translator}对象
     * @return
     */
    public Translator<F, INDArray> indexMake() {
        List<FLPair<F, INDArray>> resData = IndexMake.indexMake(data);
        return new Translator<>(resData);
    }

    /**
     * 调用了{@link IndexMake#indexMake(List, List)}生成新的数据集 返回由新数据集初始化而成的{@link Translator}对象
     * @param index2Label
     * @return
     */
    public Translator<F, INDArray> indexMake(List<L> index2Label) {
        List<FLPair<F, INDArray>> resData = IndexMake.indexMake(data,index2Label);
        return new Translator<>(resData);
    }

    /**
     * 调用了{@link IndexMake#indexMake(List, Map)}生成新的数据集 返回由新数据集初始化而成的{@link Translator}对象
     * @param label2Index
     * @return
     */
    public Translator<F, INDArray> indexMake(Map<L, Integer> label2Index) {
        List<FLPair<F, INDArray>> resData = IndexMake.indexMake(data,label2Index);
        return new Translator<>(resData);
    }

    public Translator<F, L> setData(List<FLPair<F, L>> data) {
        this.data = data;
        return this;
    }

    public List<FLPair<F, L>> getData() {
        return data;
    }
}
