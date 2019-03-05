# LibDL

## 2019.03.05
更新了```LibDL.Tensor.Operator```包中的三个类 ```L2Regularize L1Regularize View```

更新了 ```LibDL.ND4JUtil``` 类中的```public static INDArray Abs(INDArray x)```方法

```L1Regularize```类中的后向传播实现的比较繁琐 因为暂时没有找到Nd4j原生的实现方法

以上代码暂未测试