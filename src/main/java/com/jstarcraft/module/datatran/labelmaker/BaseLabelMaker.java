package com.jstarcraft.module.datatran.labelmaker;

import java.util.ArrayList;

public abstract class BaseLabelMaker<L> implements LabelMaker<L> {

    @Override
    public ArrayList<L> makeAll(Object features) {
        ArrayList<?> files = (ArrayList)features;
        ArrayList<L> res = new ArrayList<>();
        files.stream().map(this::make).forEach(res::add);
        return res;
    }
}
