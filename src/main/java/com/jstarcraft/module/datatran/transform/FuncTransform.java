package com.jstarcraft.module.datatran.transform;

import java.util.function.Function;

public class FuncTransform<OUT> implements Transform<OUT> {
    private Function<Object,OUT> function;

    public FuncTransform(Function<Object,OUT> function) {
        this.function = function;
    }

    @Override
    public OUT tran(Object in) {
        return function.apply(in);
    }

}
