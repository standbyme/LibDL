package com.jstarcraft.module.datatran.transform;

/**
 * 储存了一种变换 且<code>IN</code>类型的对象在经过改变换后变成了<code>OUT</code>类型的对象
 * @param <IN> 原始数据的数据类型
 * @param <OUT> 目标数据的数据类型
 */
public interface Transform<IN,OUT> {
    /**
     * 变换过程
     * @param in 原始数据
     * @return 目标数据
     * @throws Exception 数据的变换过程中可能发生的异常
     */
    OUT tran(Object in) throws Exception;
}
