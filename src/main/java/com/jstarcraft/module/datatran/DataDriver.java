package com.jstarcraft.module.datatran;

import com.jstarcraft.module.datatran.transform.integrate.MultiTransform;
import com.jstarcraft.module.datatran.transform.Transform;

/**
 * <p>用于表示和加载数据的类</p>
 * <p>假如某条原始数据ori经过一系列变换trans后得到了一条新的数据data 那么我们只需要记录下ori和trans 就相当于记录了data 这也是{@link DataDriver}类的对象表示数据的方式</p>
 * @param <IN> 原始数据的数据类型
 * @param <OUT> 目标数据的数据类型
 */
public class DataDriver<IN,OUT> {
    /**
     * 原始数据
     */
    private IN oriData;
    /**
     * 对原始数据所进行的变换 经过该对象的变换后 可以得到目标数据
     */
    private MultiTransform<IN,OUT> transform = new MultiTransform<>();

    /**
     * <p>构造函数 设置了{@link DataDriver#oriData}</p>
     * <p>初始化了该对象需要进行的变换</p>
     * @param oriData 原始数据
     * @param transforms 数组中的对象 会顺序被添加到该{@link DataDriver}对象的{@link DataDriver#transform}执行列表的末尾
     */
    public DataDriver(IN oriData, Transform... transforms) {
        this.oriData = oriData;
        this.transform.add(transforms);
    }

    /**
     * 向该{@link DataDriver}对象的{@link DataDriver#transform}执行列表末尾添加一系列{@link Transform}对象
     * @param transforms 一组待添加的<code>Transform</code>对象
     */
    public void add(Transform... transforms) {
        this.transform.add(transforms);
    }

    /**
     * 获得经过了变换后的目标数据
     * @return 目标数据
     * @throws Exception 数据的变换过程中可能发生的异常
     */
    public OUT translation() throws Exception{
        return transform.tran(oriData);
    }

    public IN getOriData() {
        return oriData;
    }

    public void setOriData(IN oriData) {
        this.oriData = oriData;
    }

    public MultiTransform<IN,OUT> getTransform() {
        return transform;
    }

    public void setTransform(MultiTransform<IN,OUT> transform) {
        this.transform = transform;
    }
}
