package com.jstarcraft.module.datatran.transform.integrate;

import com.jstarcraft.module.datatran.transform.Transform;

import java.util.ArrayList;
import java.util.function.Function;

public class ForeachFuncTransform<IN,OUT> extends MultiTransform<ArrayList<IN>, ArrayList<OUT>> {

    public ForeachFuncTransform() {
    }

    public ForeachFuncTransform(Transform... transformsToAdd) {
        super(transformsToAdd);
    }

    public ForeachFuncTransform(Function<?,?>... functions) {
        super(functions);
    }

    @Override
    public ArrayList<OUT> tran(Object in) throws Exception {
        ArrayList<IN> arrayIn = verifyIn(in);
        ArrayList<Object> arrayOut = new ArrayList<>();
        for (IN each : arrayIn) {
            Object res = each;
            for (Transform transform : getTransforms())
                res = transform.tran(res);
            arrayOut.add(res);
        }
        return verifyOut(arrayOut);
    }

    @Override
    public Object tran2(ArrayList<IN> ins) {
        return null;
    }
}
