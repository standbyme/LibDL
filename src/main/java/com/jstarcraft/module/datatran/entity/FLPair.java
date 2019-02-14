package com.jstarcraft.module.datatran.entity;

import java.util.function.Consumer;
import java.util.function.Function;

/**
 * 表示一个"特征-标签"对
 * @param <F> 特征对象的类型
 * @param <L> 标签对象的类型
 */
public class FLPair<F,L> {
    /**
     * 特征
     */
    private F feature = null;
    /**
     * 标签
     */
    private L label = null;

    /**
     * <p>将类型为{@link Function}&lt;FI,FO&gt;的函数对象转换成类型为{@link Function}&lt;{@link FLPair}&lt;FI,L&gt;,{@link FLPair}&lt;FO,L&gt;&gt;的函数对象 表示对一个{@link FLPair}对象中的{@link FLPair#feature}对象做某种操作</p>
     * @param function
     * @param <FI>
     * @param <FO>
     * @param <L>
     * @return
     */
    public static <FI,FO,L> Function<FLPair<FI,L>,FLPair<FO,L>> forFeature(Function<FI,FO> function){
        return (p->new FLPair<>(function.apply(p.feature),p.label));
    }

    /**
     * <p>将类型为{@link Function}&lt;LI,LO&gt;的函数对象转换成类型为{@link Function}&lt;{@link FLPair}&lt;F,LI&gt;,{@link FLPair}&lt;F,LO&gt;&gt;的函数对象 表示对一个{@link FLPair}对象中的{@link FLPair#label}对象做某种操作</p>
     * @param function
     * @param <F>
     * @param <LI>
     * @param <LO>
     * @return
     */
    public static <F,LI,LO> Function<FLPair<F,LI>,FLPair<F,LO>> forLabel(Function<LI,LO> function){
        return (p->new FLPair<>(p.feature,function.apply(p.label)));
    }

    /**
     * 对该方法的解释同{@link FLPair#forFeature(Function)}
     * @param consumer
     * @param <F>
     * @param <L>
     * @return
     */
    public static <F,L> Consumer<FLPair<F,L>> forFeature(Consumer<F> consumer) {
        return (p->consumer.accept(p.feature));
    }

    /**
     * 对该方法的解释同{@link FLPair#forLabel(Function)}
     * @param consumer
     * @param <F>
     * @param <L>
     * @return
     */
    public static <F,L> Consumer<FLPair<F,L>> forLabel(Consumer<L> consumer) {
        return (p->consumer.accept(p.label));
    }

    /**
     * 构造函数
     */
    public FLPair() {}

    /**
     * 构造函数
     * @param feature
     * @param label
     */
    public FLPair(F feature, L label) {
        this.feature = feature;
        this.label = label;
    }

    public void setFeature(F feature) {
        this.feature = feature;
    }

    public void setLabel(L label) {
        this.label = label;
    }

    public F getFeature() {
        return feature;
    }

    public L getLabel() {
        return label;
    }
}
