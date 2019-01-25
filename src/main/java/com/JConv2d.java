package com;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;

public class JConv2d {

    public static void main(String[] args)
    {
        double[][] test= new double[][]{{3,2,1},{2,5,3},{1,1,2}};
        double[][] core= new double[][]{{2,1},{2,5}};

        INDArray result=conv2d(setINDArray(test),setINDArray(core),1);
        System.out.println(Arrays.deepToString(getArray(result)));
    }
    // 相关资料网页 https://blog.csdn.net/mrhiuser/article/details/52672824
    public static INDArray col2m(INDArray target,INDArray core)
    {
        double[][] tranData=new double[(target.rows()-core.rows()+1)*(target
                .columns()-core.columns()+1)][core.rows()*core.columns()];


        return setINDArray(tranData);
    }
    public static INDArray conv2d(INDArray target, INDArray core,int step,
                                 String type)
    {

        return null;
    }
    public static INDArray conv2d(INDArray target, INDArray core,int step)
            //对输入的矩阵通过卷积核进行卷积
    {
        double[][] t=getArray(target);
        double[][] c=getArray(core);
        trans(c);
        double[][] r=new double[t.length-c.length+1][t[0].length-c[0].length+1];
        if (t.length<c.length) return null;
        if (t[0].length<c[0].length) return null;
        for (int i=0;i<t.length-c.length+1;i+=step)
        {
            for (int j=0;j<t.length-c.length+1;j++)
            {
                double v=0f;
                for (int i1=0;i1<c.length;i1++)
                {
                    for(int j1=0;j1<c.length;j1++)
                    {
                        v =v +  t[i + i1][j + j1] * c[i1][j1];
                    }
                }
                r[i][j]=v;
            }
        }
        INDArray m=setINDArray(r);
        return m;
    }
    public static INDArray[] convn(INDArray[] targets, INDArray[]
            core,int step)
    {
        INDArray[] results=new INDArray[targets.length];
        for (int i = 0; i < targets.length; i++) {
//            INDArray Scanner = targets[i];
            results[i] = conv2d(targets[i], core[i],step);
        }
        return results;
    }
    public static double[][] getArray(INDArray m)
    {
        int x=m.rows();
        int y=m.columns();
        double[][] num=new double[x][y];
        for (int i=0;i<x;i++)
            for (int j=0;j<y;j++)
            {
                num[i][j]=m.getDouble(i,j);
            }
        return num;
    }
    public static void trans(double[][] input)
    {
        for (int i=0;i<(double)input[0].length*0.5-1;i++)
        {
            int j=input.length-i;
            double[] temp=input[i];
            input[i]=input[j];
            input[j]=temp;
        }
    }
    
    public static INDArray setINDArray(double[][] data)
    {
        INDArray m = Nd4j.create(data);
//        for (int i=0;i<data.length;i++)
//            for (int j=0;j<data[0].length;j++)
//            {
//                m.putScalar(i,j,data[i][j]);
//            }
        return m;
    }




}
