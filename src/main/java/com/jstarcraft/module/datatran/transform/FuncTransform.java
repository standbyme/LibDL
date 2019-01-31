package com.jstarcraft.module.datatran.transform;

import java.util.function.Function;

/**
 * 实现了{@link BaseTransform}类 该类的对象对原始数据的变换 用一个<code>Function&lt;IN,OUT&gt;</code>类的对象来存储
 * @param <IN>
 * @param <OUT>
 */
public class FuncTransform<IN,OUT> extends BaseTransform<IN,OUT> {
    /**
     * 函数对象 存储了对原始数据的变换方法
     */
    private Function<IN,OUT> function;

    public FuncTransform(Function<IN,OUT> function) {
        this.function = function;
    }

    @Override
    public Object tran2(IN in) {
        return function.apply(in);
    }

    public void setFunction(Function<IN, OUT> function) {
        this.function = function;
    }

    public Function<IN, OUT> getFunction() {
        return function;
    }
}
