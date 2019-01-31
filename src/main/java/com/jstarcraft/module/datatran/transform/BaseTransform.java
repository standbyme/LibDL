package com.jstarcraft.module.datatran.transform;

/**
 * 指定了一种对{@link Transform}接口的实现格式
 * @param <IN> 原始数据的数据类型
 * @param <OUT> 目标数据的数据类型
 */
public abstract class BaseTransform<IN,OUT> implements Transform<IN,OUT> {
    /**
     * 变换的主要过程 供子类继承实现
     * @param in 通过了数据格式验证后的原始数据
     * @return 目标数据
     * @throws Exception 数据的变换过程中可能发生的异常
     */
    public abstract Object tran2(IN in) throws Exception;

    /**
     * <p>变换过程 包含以下三步</p>
     * <ol><li>原始数据数据格式验证</li></ol><ol><li>数据变换</li></ol><ol><li>目标数据格式验证</li></ol>
     * @param in 原始数据
     * @return 目标数据
     * @throws Exception 数据的变换过程中可能发生的异常
     */
    @Override
    public OUT tran(Object in) throws Exception {
        return verifyOut(tran2(verifyIn(in)));
    }

    /**
     * <p>原始数据格式验证 默认为强制类型转换 如果验证通过 返回<code>IN</code>类型的数据 如果不通过 抛出异常</p>
     * <p>通过对该方法的重写 不但可以验证数据所属的类 还可以验证数据的格式</p>
     * @param in 原始数据
     * @return 通过验证后原始数据
     * @throws ClassCastException 验证失败时抛出的异常
     */
    @SuppressWarnings("unchecked")
    public IN verifyIn(Object in) throws ClassCastException {
        return (IN)in;
    }

    /**
     * 目标数据格式验证 可以参照我对{@link BaseTransform#verifyIn(Object)}方法的解释
     * @param out
     * @return
     * @throws ClassCastException
     */
    @SuppressWarnings("unchecked")
    public OUT verifyOut(Object out) throws ClassCastException {
        return (OUT)out;
    }

}
