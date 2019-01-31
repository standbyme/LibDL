package com.jstarcraft.module.datatran.labelmaker;

import java.util.function.Function;

/**
 * <p>该类中有Java函数对象{@link FuncLabelMaker#function} 对{@link FuncLabelMaker#function}进行定义</p>
 * <p>然后这个{@link FuncLabelMaker}<code>&lt;F,L&gt;</code>对象 它的{@link FuncLabelMaker#make(Object)}的内容即为执行{@link FuncLabelMaker#function}</p>
 * @param <F>
 * @param <L>
 */
public class FuncLabelMaker<F,L> extends BaseLabelMaker<F,L> {
    private Function<F,L> function;

    public FuncLabelMaker(Function<F,L> function) {
        this.function = function;
    }

    @Override
    public L make(F feature) {
        return function.apply(feature);
    }

    public void setFunction(Function<F, L> function) {
        this.function = function;
    }

    public Function<F, L> getFunction() {
        return function;
    }
}
