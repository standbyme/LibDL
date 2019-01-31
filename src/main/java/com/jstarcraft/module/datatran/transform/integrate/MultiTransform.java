package com.jstarcraft.module.datatran.transform.integrate;

import com.jstarcraft.module.datatran.transform.BaseTransform;
import com.jstarcraft.module.datatran.transform.FuncTransform;
import com.jstarcraft.module.datatran.transform.Transform;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.function.Function;

public class MultiTransform<IN,OUT> extends BaseTransform<IN,OUT> {
    private ArrayList<Transform> transforms = new ArrayList<>();

    public MultiTransform() {
    }

    public MultiTransform(Transform... transformsToAdd) {
        add(transformsToAdd);
    }

    public MultiTransform(Function<?,?>... functions) {
        add(functions);
    }

    @Override
    public Object tran2(IN in) throws Exception {
        Object res = in;
        for (Transform transform : transforms)
            res = transform.tran(res);
        return res;
    }

    public void add(Transform... transformsToAdd){
        transforms.addAll(Arrays.asList(transformsToAdd));
    }

    @SuppressWarnings("unchecked")
    public void add(Function<?,?>... functions) {
        for (Function function : functions)add(new FuncTransform(function));
    }

    protected ArrayList<Transform> getTransforms() {
        return transforms;
    }
}
