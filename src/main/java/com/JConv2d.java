package com;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;

public class JConv2d {

    @Test
    public void test1()
    {
        double[][] test= new double[100][100];
        double[][] core= new double[10][10];
        //for (int i=0;i<=100;i++) {
        INDArray result1 = conv2d(setINDArray(test), setINDArray(core), 1, "origin");
        //}
    }

    @Test
    public void test2()
    {
        double[][] test= new double[100][100];
        double[][] core= new double[10][10];
        //for (int i=0;i<=100;i++) {
        INDArray result1 = conv2d(setINDArray(test), setINDArray(core), 1, "sliding");
        //}
    }

    @Test
    public void test_main()
    {
        double[][] test= new double[][]{{0,1,2,3},{4,5,6,7},{8,9,10,11},{12,13,14,15}};
        double[][] core= new double[][]{{0,1},{2,3}};
        System.out.println("Target:"+Arrays.deepToString(test));
        INDArray result1=conv2dR(setINDArray(test),1,"sliding");
        System.out.println(Arrays.deepToString(getArray(result1)));
     //   INDArray result1=conv2d(setINDArray(test),setINDArray(core),1,"origin");
        INDArray result2=conv2d(setINDArray(test),setINDArray(core),1, "sliding");
 //       INDArray result=conv2d(setINDArray(test),setINDArray(core),1);
//        assert result != null;
//        result=result.mmul(result);
    }

    // 相关资料网页 https://blog.csdn.net/mrhiuser/article/details/52672824
    public static INDArray col2im(INDArray target, int step) {
        double[][] targetData = getArray(target);
        int x = (int) Math.sqrt(targetData.length);
        int y = (int) Math.sqrt(targetData[0].length);
        double[][] tranData = new double[x + y - 1][x + y - 1];
        for (int i = 0; i < targetData.length; i+=step) {
            for (int j = 0; j < targetData[0].length; j+=step) {
                try {
                    tranData[j / x + i / y][j % x + i % y] += targetData[i][j];
                } catch (Exception ignored) {

                }
            }
        }
        return setINDArray(tranData);
    }
    public static INDArray im2col(INDArray target,INDArray core,int step)
    {
        String type="sliding";
        double[][] tranData = new double[(target.rows() - core.rows() + 1) * (target
                .columns() - core.columns() + 1)][core.rows() * core.columns()];
        double[][] targetData=getArray(target);
        if (type.equals("distinct")) {
            double[][] targetClone=new double[targetData.length+targetData
                    .length%core.rows()][targetData[0].length+targetData[0]
                    .length%core.columns()];
            for(int i=0;i<targetData.length;i++)
                for(int j=0;j<targetData[0].length;j++)
                {
                    targetClone[i][j]=targetData[i][j];
                }
            targetData=targetClone;
            System.out.println("CloneResult:"+ Arrays.deepToString(targetData));
        }
       // if (type.equals("sliding"))
        {
            int x1=target.rows() - core.rows()+1;
            int x2=core.rows();

            //在这里进行滑动操作
//            System.out.println(x1+"_"+x2);
            //滑动操作的起点x坐标

//            if (step>1)
//            {
//                // 对于滑动步长非整数倍的情况，外圈补0 TODO 该步骤暂时存疑资料不足
//                tranData=new double[(target.rows() - core.rows() + 1) * (target
//                    .columns() - core.columns() + 1)+(target.rows() - core
//                    .rows())%step][core.rows() * core.columns()+(target.columns() -
//                        core.columns())%step];
//            }

            for (int j=0;j<tranData[0].length;j+=step) {
                for (int i = 0; i < tranData.length; i+=step) {
//                    System.out.print(i+"-"+j+":");
//                    System.out.println((j/x1+i/x2)+"_"+(j%x1+i%x2));
                    try {
                        tranData[i][j] = targetData[j / x1 + i / x2][j % x1 + i % x2];
                    }
                    catch (Exception e)
                    {
                        tranData[i][j]=0;//自动补0
                    }
                }
            }
        }

        System.out.println("TargetTran:"+ Arrays.deepToString(tranData));
        return setINDArray(tranData);
    }
    public static INDArray im2colc(INDArray core)
    {
        double[][] coreData=getArray(core);
        double[][] tranCore=new double[1][coreData.length*coreData[0].length];
        int index=0;
        for (double[] Scanner:coreData)
        {
            for (double Scanner2:Scanner)
            {
                tranCore[0][index]=Scanner2;
                index++;
            }
        }
        System.out.println("Core:"+Arrays.deepToString(tranCore));
        return setINDArray(tranCore);
    }

    public static INDArray conv2d(INDArray target, INDArray core,int step, String type)
    {
//        if (type.equals("origin"))
//        {
//            return conv2d(target,core,step);
//        }
        INDArray tranTarget=im2col(target,core,step);
        INDArray tranCore=im2colc(core);
        INDArray result=tranTarget.mul(tranCore);
        //INDArray result=tranCore.mul(target);
        System.out.println("MulResult:"+Arrays.deepToString(getArray(result)));
        //return result;
        //对矩阵相乘的结果进行处理？ 不需要可直接注释下方代码
        double[][] resultData=getArray(result);
        double[][] dealData=new double[target.rows()-core.rows()+1][target.columns()-core.columns()+1];
        for (int j = 0; j < resultData[0].length; j++) {
            double sum=0;
            for (int i=0;i<resultData.length;i++ )
            {
                sum+=resultData[i][j];
            }
            dealData[j/dealData[0].length][j%dealData.length]=sum;
        }
        System.out.println("Result:"+Arrays.deepToString(dealData));
        return setINDArray(dealData);
    }

    public static INDArray conv2dR(INDArray target,int step, String type)
    {
        INDArray result=col2im(target,step);
        //INDArray result=tranCore.mul(target);
        System.out.println("Result:"+Arrays.deepToString(getArray(result)));
        return result;
    }
    public static double sum(double... input)
    {
        double sum=0;
        for (double Scanner:input) sum+=Scanner;
        return sum;
    }

//    public static INDArray conv2d(INDArray target, INDArray core,int step)
//            //对输入的矩阵通过卷积核进行卷积
//    {
//        double[][] t=getArray(target);
//        double[][] c=getArray(core);
//        trans(c);
//        double[][] r=new double[t.length-c.length+1][t[0].length-c[0].length+1];
//        if (t.length<c.length) return null;
//        if (t[0].length<c[0].length) return null;
//        for (int i=0;i<t.length-c.length+1;i+=step)
//        {
//            for (int j=0;j<t.length-c.length+1;j++)
//            {
//                double v=0f;
//                for (int i1=0;i1<c.length;i1++)
//                {
//                    for(int j1=0;j1<c.length;j1++)
//                    {
//                        v =v +  t[i + i1][j + j1] * c[i1][j1];
//                    }
//                }
//                r[i][j]=v;
//            }
//        }
//        System.out.println("Result:"+Arrays.deepToString(r));
//        return setINDArray(r);
//    }

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
            try {
                int j = input.length - i;
                double[] temp = input[i];
                input[i] = input[j];
                input[j] = temp;
            }catch (Exception ignored)
            {

            }
        }
    }
    
    public static INDArray setINDArray(double[][] data)
    {
        //        for (int i=0;i<data.length;i++)
//            for (int j=0;j<data[0].length;j++)
//            {
//                m.putScalar(i,j,data[i][j]);
//            }
        return Nd4j.create(data);
    }
//    public static INDArray deal_col2R(INDArray input)
//    {
//
//    }




}
