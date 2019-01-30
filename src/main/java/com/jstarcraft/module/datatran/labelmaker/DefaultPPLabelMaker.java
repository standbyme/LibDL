package com.jstarcraft.module.datatran.labelmaker;

public class DefaultPPLabelMaker extends ParentPathLabelMaker<String> {
    @Override
    public String classTran(String path) {
        return path;
    }
}
