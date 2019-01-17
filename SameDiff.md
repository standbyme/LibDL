# SameDiff

SameDiff Examples

SameDiff 是建立在ND4J之上的自动微分库

1.0.0-beta3中，SameDiff有一个限制，将在接下来的几个版本中修复：SameDiff中的许多操作仅限于在CPU中执行。未来将放在GPU中（CUDA）执行

SameDiff可以创建DL4J的Layer和Vertix作为MultiLayerNetwork或ComputationGraph的一部分

SameDiff支持从TensorFlow模型创建SameDiff图。此功能尚不完善 - 某些操作不可用。

Java 11 暂不可用


# Ex1_SameDiff_Basics

**SRC**

```java
package org.nd4j.samediff;

import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;
import java.util.List;

/**
 * This is a simple example introducing the basic concepts of SameDiff:
 * - The SameDiff class
 * - Variables
 * - Functions
 * - Getting arrays
 * - Executing forward pass
 *
 * @author Alex Black
 */
public class Ex1_SameDiff_Basics {

    public static void main(String[] args) {


        //The starting point for using SameDiff is creating a SameDiff instance via the create method.
        //The SameDiff class has many methods for creating variables and functions
        SameDiff sd = SameDiff.create();

        /*
        Variables can be added to a graph in a number of ways.
        You can think of variables as "holders" of an n-dimensional array - specifically, an ND4J INDArray
        First, let's create a variable based on a specified array, with the name "myVariable"
        */
        INDArray values = Nd4j.ones(3,4);
        SDVariable variable = sd.var("myVariable", values);

        //We can then perform operations on the variable:
        SDVariable plusOne = variable.add(1.0);                               //Name: automatically generated as "add"
        SDVariable mulTen = variable.mul("mulTen", 10.0);        //Name: Defined to be "mulTen"
        //Note that most operations have method overloads with and without a user-specified name - i.e., add(double) vs. add(String,double)

        //Let's inspect the graph. We currently have 3 variables, with names "myVariable", "add", and "mulTen",
        // and two functions - our "add 1.0" and our "multiply by 10" functions. These are shown in the summary:
        System.out.println(sd.summary());
        System.out.println("===================================");

        //We can also get the variables directly:
        List<SDVariable> allVariables = sd.variables();
        System.out.println("Variables: " + allVariables);
        for(SDVariable var : allVariables){
            long[] varShape = var.getShape();
            System.out.println(var.getVarName() + " - shape " + Arrays.toString(varShape));
        }

        /*
        We can also inspect the individual functions.
        You can think of functions as operations that map input variables (arrays) to new variables (arrays)
        Functions have 0 or more inputs (usually 1 or more) and 1 or more outputs
         */
        DifferentialFunction[] functions = sd.functions();
        for(DifferentialFunction df : functions){
            SDVariable[] inputsToFunction = df.args();      //Inputs are also known as "args" or "arguments" for a function
            SDVariable[] outputsOfFunction = df.outputVariables();
            System.out.println("Op: " + df.opName() + ", inputs: " + Arrays.toString(inputsToFunction) + ", outputs: " +
                Arrays.toString(outputsOfFunction));
        }


        //Now, let's execute the graph forward pass:
        sd.exec();

        INDArray variableArr = variable.getArr();               //We can get arrays directly from the variables
        INDArray plusOneArr = plusOne.getArr();
        INDArray mulTenArr = sd.getArrForVarName("mulTen");     //Or also by name, from the Samediff instance

        System.out.println("===================================");
        System.out.println("Initial variable values:\n" + variableArr);
        System.out.println("'plusOne' values:\n" + plusOneArr);
        System.out.println("'mulTen' values:\n" + mulTenArr);



        //That's it - see the next example for calculating gradients
    }

}

```

**OUTPUT**

```
o.n.l.f.Nd4jBackend - Loaded [CpuBackend] backend
o.n.n.NativeOpsHolder - Number of threads used for NativeOps: 2
o.n.n.Nd4jBlas - Number of threads used for BLAS: 2
o.n.l.a.o.e.DefaultOpExecutioner - Backend used: [CPU]; OS: [Linux]
o.n.l.a.o.e.DefaultOpExecutioner - Cores: [4]; Memory: [1.7GB];
o.n.l.a.o.e.DefaultOpExecutioner - Blas vendor: [MKL]
--- Summary ---
Variables:               3                    (3 with arrays)
Functions:               2                   
SameDiff Function Defs:  0                   

--- Variables ---
- Name -    - Array Shape -     - Output Of Function -  - Inputs To Functions -
myVariable  [3, 4]              <none>                  [add_scalar, mul_scalar]
add         [3, 4]              add_scalar(add_scalar)                      
mulTen      [3, 4]              mul_scalar(mul_scalar)                      


--- Functions ---
     - Function Name -  - Op -                - Inputs -    - Outputs -  
0    add_scalar         ScalarAdd             [myVariable]  [add]        
1    mul_scalar         ScalarMultiplication  [myVariable]  [mulTen]     

===================================
Variables: [myVariable, add, mulTen]
myVariable - shape [3, 4]
add - shape [3, 4]
mulTen - shape [3, 4]
Op: add_scalar, inputs: [myVariable], outputs: [add]
Op: mul_scalar, inputs: [myVariable], outputs: [mulTen]
===================================
Initial variable values:
[[    1.0000,    1.0000,    1.0000,    1.0000], 
 [    1.0000,    1.0000,    1.0000,    1.0000], 
 [    1.0000,    1.0000,    1.0000,    1.0000]]
'plusOne' values:
[[    2.0000,    2.0000,    2.0000,    2.0000], 
 [    2.0000,    2.0000,    2.0000,    2.0000], 
 [    2.0000,    2.0000,    2.0000,    2.0000]]
'mulTen' values:
[[   10.0000,   10.0000,   10.0000,   10.0000], 
 [   10.0000,   10.0000,   10.0000,   10.0000], 
 [   10.0000,   10.0000,   10.0000,   10.0000]]
```

# Ex2_LinearRegression.java

**SRC**

```java
package org.nd4j.samediff;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.weightinit.impl.XavierInitScheme;

/**
 * This example shows how to implement a simple linear regression graph with a mean-squared error loss function.
 *
 * Specifically, we will implement:
 * - output = input * weights + bias
 * - loss = MSE(output, label) = 1/(nExamples * nOut) * sum_i (labels_i - out_i) ^ 2
 *
 * We will have:
 * nIn = 4
 * nOut = 1
 *
 * @author Alex Black
 */
public class Ex2_LinearRegression {

    public static void main(String[] args) {
        //How to calculate gradients, and get gradient arrays - linear regression (MSE, manually defined)

        int nIn = 4;
        //Why there is 2 not 1? I've opened issue.
        int nOut = 2;

        SameDiff sd = SameDiff.create();

        //First: let's create our variables
        //The second arg in sd.var is shape
        //-1 means placeholder just like numpy
        SDVariable input = sd.var("input", new long[]{-1, nIn});
        //Why there is {-1, 1} not {-1, 2} ? I've opened issue.
        SDVariable labels = sd.var("labels", new long[]{-1, 1});
        SDVariable weights = sd.var("weights", new long[]{nIn,nOut}, new XavierInitScheme('c', nIn, nOut));
        SDVariable bias = sd.var("bias");


        //And define our forward pass:
        SDVariable out = input.mmul(weights).add(bias);     //Note: it's broadcast add here

        //And our loss function (done manually here for the purposes of this example):
        SDVariable difference = labels.sub(out);
        SDVariable sqDiff = sd.square(difference);
        SDVariable mse = sqDiff.mean();

        //Let's create some mock data for this example:
        int minibatch = 3;
        Nd4j.getRandom().setSeed(12345);
        INDArray inputArr = Nd4j.rand(minibatch, nIn);
        INDArray labelArr = Nd4j.rand(minibatch, nOut);

        //Associate these variables with the SameDiff instance:
        input.setArray(inputArr);                                   //One approach
        sd.associateArrayWithVariable(labelArr, "labels");   //Alternative but equivalent

        //Execute forward pass:
        INDArray loss = sd.execAndEndResult();
        System.out.println("MSE: " + loss);

        //Calculate gradients:
        sd.execBackwards();

        //Get gradients for each variable:
        for(SDVariable v : new SDVariable[]{weights, bias}){
            System.out.println("Variable name: " + v.getVarName());
            System.out.println("Values:\n" + v.getArr());
            System.out.println("Gradients:\n" + v.getGradient().getArr());
        }
    }

}
```

**OUTPUT**

```
o.n.l.f.Nd4jBackend - Loaded [CpuBackend] backend
o.n.n.NativeOpsHolder - Number of threads used for NativeOps: 2
o.n.n.Nd4jBlas - Number of threads used for BLAS: 2
o.n.l.a.o.e.DefaultOpExecutioner - Backend used: [CPU]; OS: [Linux]
o.n.l.a.o.e.DefaultOpExecutioner - Cores: [4]; Memory: [1.7GB];
o.n.l.a.o.e.DefaultOpExecutioner - Blas vendor: [MKL]
o.n.l.a.o.DynamicCustomOp - No input found for add and op name subtract
MSE: 0.8508
Variable name: weights
Values:
[[   -0.1984,   -0.2110], 
 [    0.3990,    0.8112], 
 [   -0.8519,    1.0251], 
 [    0.8928,   -0.2008]]
Gradients:
[[   -0.2087,    0.1213], 
 [   -0.3092,    0.2445], 
 [   -0.6247,    0.5990], 
 [   -0.2056,    0.2174]]
Variable name: bias
Values:
0
Gradients:
-0.3093
```

# Ex3_Variables.java

**SRC**

```java
package org.nd4j.samediff;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.distribution.impl.TruncatedNormalDistribution;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.weightinit.WeightInitScheme;
import org.nd4j.weightinit.impl.DistributionInitScheme;

/**
 * This example: shows some more ways to create variables
 *
 * @author Alex Black
 */
public class Ex3_Variables {

    public static void main(String[] args) {

        SameDiff sd = SameDiff.create();

        //There are multiple ways to create the initial variables in your graph
        // First, you can create them directly, with the specified array:
        INDArray arr = Nd4j.arange(10);
        SDVariable var1 = sd.var("var1", arr);

        /*
        Or create a variable by specifying a shape and an initializer.
        The idea with an initializer is that is specifies how the initial value should be created.
        This initialization is performed once only, when the array is first required.

        -- Available Initialization Schemes --
        ContantInitScheme
        DistributionInitScheme
        IdentityInitScheme
        LecunInitScheme
        OneInitScheme
        ReluInitScheme
        ReluUniformInitScheme
        SigmoidInitScheme
        UniformInitScheme
        VarScalingNormalFanAvgInitScheme
        VarScalingNormalFanInInitScheme
        VarScalingNormalFanOutInitScheme
        VarScalingNormalUniformFanInInitScheme
        VarScalingNormalUniformFanOutInitScheme
        VarScalingUniformFanAvgInitScheme
        XavierFanInInitScheme
        XavierInitScheme
        XavierUniformInitScheme
        ZeroInitScheme
         */
        long[] shape = new long[]{3,4};
        WeightInitScheme initScheme = new DistributionInitScheme(Nd4j.order(), new TruncatedNormalDistribution(0, 1));
        SDVariable var2 = sd.var("var2", shape, initScheme);

        //Note that the array will be allocated using the specified distribution (truncated normal), when we try to get the array:
        INDArray var2Array = var2.getArr();
        System.out.println("var2 array values:\n" + var2Array);

        //Alternatively, we can simply specify an shape. This will default to a zero initialization for the array (if required)
        // or you can set the array directly
        SDVariable var3 = sd.var("var3", 3,4);

        INDArray values = Nd4j.ones(3,4);
        var3.setArray(values);
        System.out.println("var3 array values:\n" + var3.getArr());


        //Note also that there are a number of functions that can be used to create variables.
        //However, unlike the WeightInitScheme of the variables earlier, the random values here will be re-generated
        // on every forward pass
        SDVariable scalar = sd.scalar("scalar", 0.5);
        SDVariable zero = sd.zero("zero", new long[]{3,4});
        SDVariable zeroToNine = sd.linspace("zeroToNine", 0, 9, 10);
        SDVariable randomUniform = sd.randomUniform(-1, 1, 3,4);      //-1 to 1, shape [3,4]
        SDVariable randomBernoulli = sd.randomBernoulli(0.5, 3,4);          //Random Bernoulli: 0 or 1 with probability 0.5
        sd.exec();
        
        System.out.println("scalar array values:\n" + scalar.getArr());
        System.out.println("zero array values:\n" + zero.getArr());
        System.out.println("zeroToNine array values:\n" + zeroToNine.getArr());
        System.out.println("randomUniform array values:\n" + randomUniform.getArr());
        System.out.println("randomBernoulli array values:\n" + randomBernoulli.getArr());
    }
}
```

**OUTPUT**

```
o.n.l.f.Nd4jBackend - Loaded [CpuBackend] backend
o.n.n.NativeOpsHolder - Number of threads used for NativeOps: 2
o.n.n.Nd4jBlas - Number of threads used for BLAS: 2
o.n.l.a.o.e.DefaultOpExecutioner - Backend used: [CPU]; OS: [Linux]
o.n.l.a.o.e.DefaultOpExecutioner - Cores: [4]; Memory: [1.7GB];
o.n.l.a.o.e.DefaultOpExecutioner - Blas vendor: [MKL]
var2 array values:
[[   -0.1782,    0.2836,   -1.8047,   -1.2454], 
 [    0.7043,    1.2561,    0.2644,   -1.3786], 
 [   -0.2952,   -0.2597,    0.6272,    0.8629]]
var3 array values:
[[    1.0000,    1.0000,    1.0000,    1.0000], 
 [    1.0000,    1.0000,    1.0000,    1.0000], 
 [    1.0000,    1.0000,    1.0000,    1.0000]]
scalar array values:
0.5000
zero array values:
[[         0,         0,         0,         0], 
 [         0,         0,         0,         0], 
 [         0,         0,         0,         0]]
zeroToNine array values:
[         0,    1.0000,    2.0000,    3.0000,    4.0000,    5.0000,    6.0000,    7.0000,    8.0000,    9.0000]
randomUniform array values:
[[    0.6745,   -0.9481,   -0.3876,   -0.5322], 
 [   -0.0917,   -0.4617,   -0.4933,   -0.7506], 
 [   -0.0931,   -0.1209,    0.8544,    0.4290]]
randomBernoulli array values:
[[         0,    1.0000,         0,    1.0000], 
 [         0,         0,    1.0000,         0], 
 [         0,         0,         0,    1.0000]]
```