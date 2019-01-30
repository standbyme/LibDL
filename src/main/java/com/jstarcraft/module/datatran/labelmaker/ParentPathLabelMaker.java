package com.jstarcraft.module.datatran.labelmaker;

import java.io.File;

public abstract class ParentPathLabelMaker<L> extends BaseLabelMaker<L>{

    @Override
    public L make(Object feature) {
        File file = (File)feature;
        return classTran(file.getParentFile().getName());
    }

    public abstract L classTran(String path);
}
