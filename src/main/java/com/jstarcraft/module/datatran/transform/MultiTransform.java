package com.jstarcraft.module.datatran.transform;

import java.util.ArrayList;
import java.util.Arrays;

public class MultiTransform<OUT> implements Transform<OUT> {
    private ArrayList<Transform> transforms = new ArrayList<>();

    public MultiTransform(Transform... transformsToAdd) {
        add(transformsToAdd);
    }

    @SuppressWarnings("unchecked")
    @Override
    public OUT tran(Object in) throws Exception{
        Object res = in;
        for (Transform transform : transforms) res = transform.tran(res);
        return (OUT)res;
    }

    public void add(Transform... transformsToAdd){
        transforms.addAll(Arrays.asList(transformsToAdd));
    }

    protected ArrayList<Transform> getTransforms() {
        return transforms;
    }
}
