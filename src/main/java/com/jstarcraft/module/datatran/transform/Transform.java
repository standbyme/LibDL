package com.jstarcraft.module.datatran.transform;

public interface Transform<OUT> {
    OUT tran(Object in) throws Exception;
}
