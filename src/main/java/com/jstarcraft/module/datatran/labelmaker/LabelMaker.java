package com.jstarcraft.module.datatran.labelmaker;

import java.util.ArrayList;

public interface LabelMaker<L> {
    L make(Object feature);
    ArrayList<L> makeAll(Object features);
}
