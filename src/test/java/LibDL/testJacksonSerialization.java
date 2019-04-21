package LibDL;


import LibDL.Tensor.Constant;
import LibDL.Tensor.Tensor;
import LibDL.nn.Linear;
import LibDL.nn.SoftmaxWithLoss;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.File;
import java.io.IOException;

public class testJacksonSerialization {

    public static void main(String [] args) {
//        Constant testConstant = new Constant(new INDArray() {
//            @Override
//            public String shapeInfoToString() {
//                return null;
//            }
//
//            @Override
//            public DataBuffer shapeInfoDataBuffer() {
//                return null;
//            }
//
//            @Override
//            public DataBuffer sparseInfoDataBuffer() {
//                return null;
//            }
//
//            @Override
//            public LongBuffer shapeInfo() {
//                return null;
//            }
//
//            @Override
//            public boolean isView() {
//                return false;
//            }
//
//            @Override
//            public boolean isSparse() {
//                return false;
//            }
//
//            @Override
//            public boolean isCompressed() {
//                return false;
//            }
//
//            @Override
//            public void markAsCompressed(boolean reallyCompressed) {
//
//            }
//
//            @Override
//            public void setWrapAround(boolean wrapAround) {
//
//            }
//
//            @Override
//            public boolean isWrapAround() {
//                return false;
//            }
//
//            @Override
//            public int rank() {
//                return 0;
//            }
//
//            @Override
//            public int stride(int dimension) {
//                return 0;
//            }
//
//            @Override
//            public int elementStride() {
//                return 0;
//            }
//
//            @Override
//            public int elementWiseStride() {
//                return 0;
//            }
//
//            @Override
//            public boolean isCleanedUp() {
//                return false;
//            }
//
//            @Override
//            public void cleanup() {
//
//            }
//
//            @Override
//            public void resetLinearView() {
//
//            }
//
//            @Override
//            public int secondaryStride() {
//                return 0;
//            }
//
//            @Override
//            public double getDoubleUnsafe(long offset) {
//                return 0;
//            }
//
//            @Override
//            public INDArray putScalarUnsafe(long offset, double value) {
//                return null;
//            }
//
//            @Override
//            public int majorStride() {
//                return 0;
//            }
//
//            @Override
//            public int innerMostStride() {
//                return 0;
//            }
//
//            @Override
//            public INDArray linearView() {
//                return null;
//            }
//
//            @Override
//            public INDArray linearViewColumnOrder() {
//                return null;
//            }
//
//            @Override
//            public long vectorsAlongDimension(int dimension) {
//                return 0;
//            }
//
//            @Override
//            public INDArray vectorAlongDimension(int index, int dimension) {
//                return null;
//            }
//
//            @Override
//            public long tensorssAlongDimension(int... dimension) {
//                return 0;
//            }
//
//            @Override
//            public INDArray tensorAlongDimension(int index, int... dimension) {
//                return null;
//            }
//
//            @Override
//            public INDArray javaTensorAlongDimension(int index, int... dimension) {
//                return null;
//            }
//
//            @Override
//            public INDArray cumsumi(int dimension) {
//                return null;
//            }
//
//            @Override
//            public INDArray cumsum(int dimension) {
//                return null;
//            }
//
//            @Override
//            public INDArray assign(INDArray arr) {
//                return null;
//            }
//
//            @Override
//            public INDArray assignIf(INDArray arr, Condition condition) {
//                return null;
//            }
//
//            @Override
//            public INDArray replaceWhere(INDArray arr, Condition condition) {
//                return null;
//            }
//
//            @Override
//            public INDArray putScalar(long i, double value) {
//                return null;
//            }
//
//            @Override
//            public INDArray putScalar(long i, float value) {
//                return null;
//            }
//
//            @Override
//            public INDArray putScalar(long i, int value) {
//                return null;
//            }
//
//            @Override
//            public INDArray putScalar(int[] i, double value) {
//                return null;
//            }
//
//            @Override
//            public INDArray putScalar(long[] i, double value) {
//                return null;
//            }
//
//            @Override
//            public INDArray putScalar(long[] i, float value) {
//                return null;
//            }
//
//            @Override
//            public INDArray putScalar(long[] i, int value) {
//                return null;
//            }
//
//            @Override
//            public INDArray putScalar(long row, long col, double value) {
//                return null;
//            }
//
//            @Override
//            public INDArray putScalar(long dim0, long dim1, long dim2, double value) {
//                return null;
//            }
//
//            @Override
//            public INDArray putScalar(long dim0, long dim1, long dim2, long dim3, double value) {
//                return null;
//            }
//
//            @Override
//            public INDArray lt(Number other) {
//                return null;
//            }
//
//            @Override
//            public INDArray lti(Number other) {
//                return null;
//            }
//
//            @Override
//            public INDArray putScalar(int[] indexes, float value) {
//                return null;
//            }
//
//            @Override
//            public INDArray putScalar(int[] indexes, int value) {
//                return null;
//            }
//
//            @Override
//            public INDArray eps(Number other) {
//                return null;
//            }
//
//            @Override
//            public INDArray epsi(Number other) {
//                return null;
//            }
//
//            @Override
//            public INDArray eq(Number other) {
//                return null;
//            }
//
//            @Override
//            public INDArray eqi(Number other) {
//                return null;
//            }
//
//            @Override
//            public INDArray gt(Number other) {
//                return null;
//            }
//
//            @Override
//            public INDArray gte(Number other) {
//                return null;
//            }
//
//            @Override
//            public INDArray lte(Number other) {
//                return null;
//            }
//
//            @Override
//            public INDArray gtei(Number other) {
//                return null;
//            }
//
//            @Override
//            public INDArray ltei(Number other) {
//                return null;
//            }
//
//            @Override
//            public INDArray gti(Number other) {
//                return null;
//            }
//
//            @Override
//            public INDArray lt(INDArray other) {
//                return null;
//            }
//
//            @Override
//            public INDArray lti(INDArray other) {
//                return null;
//            }
//
//            @Override
//            public INDArray eps(INDArray other) {
//                return null;
//            }
//
//            @Override
//            public INDArray epsi(INDArray other) {
//                return null;
//            }
//
//            @Override
//            public INDArray neq(Number other) {
//                return null;
//            }
//
//            @Override
//            public INDArray neqi(Number other) {
//                return null;
//            }
//
//            @Override
//            public INDArray neq(INDArray other) {
//                return null;
//            }
//
//            @Override
//            public INDArray neqi(INDArray other) {
//                return null;
//            }
//
//            @Override
//            public INDArray eq(INDArray other) {
//                return null;
//            }
//
//            @Override
//            public INDArray eqi(INDArray other) {
//                return null;
//            }
//
//            @Override
//            public INDArray gt(INDArray other) {
//                return null;
//            }
//
//            @Override
//            public INDArray gti(INDArray other) {
//                return null;
//            }
//
//            @Override
//            public INDArray neg() {
//                return null;
//            }
//
//            @Override
//            public INDArray negi() {
//                return null;
//            }
//
//            @Override
//            public INDArray rdiv(Number n) {
//                return null;
//            }
//
//            @Override
//            public INDArray rdivi(Number n) {
//                return null;
//            }
//
//            @Override
//            public INDArray rsub(Number n) {
//                return null;
//            }
//
//            @Override
//            public INDArray rsubi(Number n) {
//                return null;
//            }
//
//            @Override
//            public INDArray div(Number n) {
//                return null;
//            }
//
//            @Override
//            public INDArray divi(Number n) {
//                return null;
//            }
//
//            @Override
//            public INDArray mul(Number n) {
//                return null;
//            }
//
//            @Override
//            public INDArray muli(Number n) {
//                return null;
//            }
//
//            @Override
//            public INDArray sub(Number n) {
//                return null;
//            }
//
//            @Override
//            public INDArray subi(Number n) {
//                return null;
//            }
//
//            @Override
//            public INDArray add(Number n) {
//                return null;
//            }
//
//            @Override
//            public INDArray addi(Number n) {
//                return null;
//            }
//
//            @Override
//            public INDArray rdiv(Number n, INDArray result) {
//                return null;
//            }
//
//            @Override
//            public INDArray rdivi(Number n, INDArray result) {
//                return null;
//            }
//
//            @Override
//            public INDArray rsub(Number n, INDArray result) {
//                return null;
//            }
//
//            @Override
//            public INDArray rsubi(Number n, INDArray result) {
//                return null;
//            }
//
//            @Override
//            public INDArray div(Number n, INDArray result) {
//                return null;
//            }
//
//            @Override
//            public INDArray divi(Number n, INDArray result) {
//                return null;
//            }
//
//            @Override
//            public INDArray mul(Number n, INDArray result) {
//                return null;
//            }
//
//            @Override
//            public INDArray muli(Number n, INDArray result) {
//                return null;
//            }
//
//            @Override
//            public INDArray sub(Number n, INDArray result) {
//                return null;
//            }
//
//            @Override
//            public INDArray subi(Number n, INDArray result) {
//                return null;
//            }
//
//            @Override
//            public INDArray add(Number n, INDArray result) {
//                return null;
//            }
//
//            @Override
//            public INDArray addi(Number n, INDArray result) {
//                return null;
//            }
//
//            @Override
//            public INDArray get(INDArrayIndex... indexes) {
//                return null;
//            }
//
//            @Override
//            public INDArray match(INDArray comp, Condition condition) {
//                return null;
//            }
//
//            @Override
//            public INDArray match(Number comp, Condition condition) {
//                return null;
//            }
//
//            @Override
//            public INDArray getWhere(INDArray comp, Condition condition) {
//                return null;
//            }
//
//            @Override
//            public INDArray getWhere(Number comp, Condition condition) {
//                return null;
//            }
//
//            @Override
//            public INDArray putWhere(INDArray comp, INDArray put, Condition condition) {
//                return null;
//            }
//
//            @Override
//            public INDArray putWhere(Number comp, INDArray put, Condition condition) {
//                return null;
//            }
//
//            @Override
//            public INDArray putWhereWithMask(INDArray mask, INDArray put) {
//                return null;
//            }
//
//            @Override
//            public INDArray putWhereWithMask(INDArray mask, Number put) {
//                return null;
//            }
//
//            @Override
//            public INDArray putWhere(Number comp, Number put, Condition condition) {
//                return null;
//            }
//
//            @Override
//            public INDArray get(INDArray indices) {
//                return null;
//            }
//
//            @Override
//            public INDArray get(List<List<Integer>> indices) {
//                return null;
//            }
//
//            @Override
//            public INDArray getColumns(int... columns) {
//                return null;
//            }
//
//            @Override
//            public INDArray getRows(int... rows) {
//                return null;
//            }
//
//            @Override
//            public INDArray rdiv(INDArray other) {
//                return null;
//            }
//
//            @Override
//            public INDArray rdivi(INDArray other) {
//                return null;
//            }
//
//            @Override
//            public INDArray rdiv(INDArray other, INDArray result) {
//                return null;
//            }
//
//            @Override
//            public INDArray rdivi(INDArray other, INDArray result) {
//                return null;
//            }
//
//            @Override
//            public INDArray rsub(INDArray other, INDArray result) {
//                return null;
//            }
//
//            @Override
//            public INDArray rsub(INDArray other) {
//                return null;
//            }
//
//            @Override
//            public INDArray rsubi(INDArray other) {
//                return null;
//            }
//
//            @Override
//            public INDArray rsubi(INDArray other, INDArray result) {
//                return null;
//            }
//
//            @Override
//            public INDArray assign(Number value) {
//                return null;
//            }
//
//            @Override
//            public long linearIndex(long i) {
//                return 0;
//            }
//
//            @Override
//            public void sliceVectors(List<INDArray> list) {
//
//            }
//
//            @Override
//            public INDArray putSlice(int slice, INDArray put) {
//                return null;
//            }
//
//            @Override
//            public INDArray cond(Condition condition) {
//                return null;
//            }
//
//            @Override
//            public INDArray condi(Condition condition) {
//                return null;
//            }
//
//            @Override
//            public INDArray repmat(int... shape) {
//                return null;
//            }
//
//            @Override
//            public INDArray repeat(int dimension, int... repeats) {
//                return null;
//            }
//
//            @Override
//            public INDArray repeat(int dimension, long... repeats) {
//                return null;
//            }
//
//            @Override
//            public INDArray putRow(long row, INDArray toPut) {
//                return null;
//            }
//
//            @Override
//            public INDArray putColumn(int column, INDArray toPut) {
//                return null;
//            }
//
//            @Override
//            public INDArray getScalar(long row, long column) {
//                return null;
//            }
//
//            @Override
//            public INDArray getScalar(long i) {
//                return null;
//            }
//
//            @Override
//            public long index(long row, long column) {
//                return 0;
//            }
//
//            @Override
//            public double squaredDistance(INDArray other) {
//                return 0;
//            }
//
//            @Override
//            public double distance2(INDArray other) {
//                return 0;
//            }
//
//            @Override
//            public double distance1(INDArray other) {
//                return 0;
//            }
//
//            @Override
//            public INDArray put(List<List<Integer>> indices, INDArray element) {
//                return null;
//            }
//
//            @Override
//            public INDArray put(INDArray indices, INDArray element) {
//                return null;
//            }
//
//            @Override
//            public INDArray put(INDArrayIndex[] indices, INDArray element) {
//                return null;
//            }
//
//            @Override
//            public INDArray put(INDArrayIndex[] indices, Number element) {
//                return null;
//            }
//
//            @Override
//            public INDArray put(int[] indices, INDArray element) {
//                return null;
//            }
//
//            @Override
//            public INDArray put(int i, int j, INDArray element) {
//                return null;
//            }
//
//            @Override
//            public INDArray put(int i, int j, Number element) {
//                return null;
//            }
//
//            @Override
//            public INDArray put(int i, INDArray element) {
//                return null;
//            }
//
//            @Override
//            public INDArray diviColumnVector(INDArray columnVector) {
//                return null;
//            }
//
//            @Override
//            public INDArray divColumnVector(INDArray columnVector) {
//                return null;
//            }
//
//            @Override
//            public INDArray diviRowVector(INDArray rowVector) {
//                return null;
//            }
//
//            @Override
//            public INDArray divRowVector(INDArray rowVector) {
//                return null;
//            }
//
//            @Override
//            public INDArray rdiviColumnVector(INDArray columnVector) {
//                return null;
//            }
//
//            @Override
//            public INDArray rdivColumnVector(INDArray columnVector) {
//                return null;
//            }
//
//            @Override
//            public INDArray rdiviRowVector(INDArray rowVector) {
//                return null;
//            }
//
//            @Override
//            public INDArray rdivRowVector(INDArray rowVector) {
//                return null;
//            }
//
//            @Override
//            public INDArray muliColumnVector(INDArray columnVector) {
//                return null;
//            }
//
//            @Override
//            public INDArray mulColumnVector(INDArray columnVector) {
//                return null;
//            }
//
//            @Override
//            public INDArray muliRowVector(INDArray rowVector) {
//                return null;
//            }
//
//            @Override
//            public INDArray mulRowVector(INDArray rowVector) {
//                return null;
//            }
//
//            @Override
//            public INDArray rsubiColumnVector(INDArray columnVector) {
//                return null;
//            }
//
//            @Override
//            public INDArray rsubColumnVector(INDArray columnVector) {
//                return null;
//            }
//
//            @Override
//            public INDArray rsubiRowVector(INDArray rowVector) {
//                return null;
//            }
//
//            @Override
//            public INDArray rsubRowVector(INDArray rowVector) {
//                return null;
//            }
//
//            @Override
//            public INDArray subiColumnVector(INDArray columnVector) {
//                return null;
//            }
//
//            @Override
//            public INDArray subColumnVector(INDArray columnVector) {
//                return null;
//            }
//
//            @Override
//            public INDArray subiRowVector(INDArray rowVector) {
//                return null;
//            }
//
//            @Override
//            public INDArray subRowVector(INDArray rowVector) {
//                return null;
//            }
//
//            @Override
//            public INDArray addiColumnVector(INDArray columnVector) {
//                return null;
//            }
//
//            @Override
//            public INDArray putiColumnVector(INDArray columnVector) {
//                return null;
//            }
//
//            @Override
//            public INDArray addColumnVector(INDArray columnVector) {
//                return null;
//            }
//
//            @Override
//            public INDArray addiRowVector(INDArray rowVector) {
//                return null;
//            }
//
//            @Override
//            public INDArray putiRowVector(INDArray rowVector) {
//                return null;
//            }
//
//            @Override
//            public INDArray addRowVector(INDArray rowVector) {
//                return null;
//            }
//
//            @Override
//            public INDArray mmul(INDArray other, MMulTranspose mMulTranspose) {
//                return null;
//            }
//
//            @Override
//            public INDArray mmul(INDArray other) {
//                return null;
//            }
//
//            @Override
//            public double[][] toDoubleMatrix() {
//                return new double[0][];
//            }
//
//            @Override
//            public double[] toDoubleVector() {
//                return new double[0];
//            }
//
//            @Override
//            public float[] toFloatVector() {
//                return new float[0];
//            }
//
//            @Override
//            public float[][] toFloatMatrix() {
//                return new float[0][];
//            }
//
//            @Override
//            public int[] toIntVector() {
//                return new int[0];
//            }
//
//            @Override
//            public long[] toLongVector() {
//                return new long[0];
//            }
//
//            @Override
//            public long[][] toLongMatrix() {
//                return new long[0][];
//            }
//
//            @Override
//            public int[][] toIntMatrix() {
//                return new int[0][];
//            }
//
//            @Override
//            public INDArray mmul(INDArray other, INDArray result) {
//                return null;
//            }
//
//            @Override
//            public INDArray mmul(INDArray other, INDArray result, MMulTranspose mMulTranspose) {
//                return null;
//            }
//
//            @Override
//            public INDArray div(INDArray other) {
//                return null;
//            }
//
//            @Override
//            public INDArray div(INDArray other, INDArray result) {
//                return null;
//            }
//
//            @Override
//            public INDArray mul(INDArray other) {
//                return null;
//            }
//
//            @Override
//            public INDArray mul(INDArray other, INDArray result) {
//                return null;
//            }
//
//            @Override
//            public INDArray sub(INDArray other) {
//                return null;
//            }
//
//            @Override
//            public INDArray sub(INDArray other, INDArray result) {
//                return null;
//            }
//
//            @Override
//            public INDArray add(INDArray other) {
//                return null;
//            }
//
//            @Override
//            public INDArray add(INDArray other, INDArray result) {
//                return null;
//            }
//
//            @Override
//            public INDArray mmuli(INDArray other, MMulTranspose transpose) {
//                return null;
//            }
//
//            @Override
//            public INDArray mmuli(INDArray other) {
//                return null;
//            }
//
//            @Override
//            public INDArray mmuli(INDArray other, INDArray result, MMulTranspose transpose) {
//                return null;
//            }
//
//            @Override
//            public INDArray mmuli(INDArray other, INDArray result) {
//                return null;
//            }
//
//            @Override
//            public INDArray divi(INDArray other) {
//                return null;
//            }
//
//            @Override
//            public INDArray divi(INDArray other, INDArray result) {
//                return null;
//            }
//
//            @Override
//            public INDArray muli(INDArray other) {
//                return null;
//            }
//
//            @Override
//            public INDArray muli(INDArray other, INDArray result) {
//                return null;
//            }
//
//            @Override
//            public INDArray subi(INDArray other) {
//                return null;
//            }
//
//            @Override
//            public INDArray subi(INDArray other, INDArray result) {
//                return null;
//            }
//
//            @Override
//            public INDArray addi(INDArray other) {
//                return null;
//            }
//
//            @Override
//            public INDArray addi(INDArray other, INDArray result) {
//                return null;
//            }
//
//            @Override
//            public INDArray normmax(int... dimension) {
//                return null;
//            }
//
//            @Override
//            public Number normmaxNumber() {
//                return null;
//            }
//
//            @Override
//            public INDArray norm2(int... dimension) {
//                return null;
//            }
//
//            @Override
//            public Number norm2Number() {
//                return null;
//            }
//
//            @Override
//            public INDArray norm1(int... dimension) {
//                return null;
//            }
//
//            @Override
//            public Number norm1Number() {
//                return null;
//            }
//
//            @Override
//            public INDArray std(int... dimension) {
//                return null;
//            }
//
//            @Override
//            public Number stdNumber() {
//                return null;
//            }
//
//            @Override
//            public INDArray std(boolean biasCorrected, int... dimension) {
//                return null;
//            }
//
//            @Override
//            public Number stdNumber(boolean biasCorrected) {
//                return null;
//            }
//
//            @Override
//            public INDArray prod(int... dimension) {
//                return null;
//            }
//
//            @Override
//            public Number prodNumber() {
//                return null;
//            }
//
//            @Override
//            public INDArray mean(int... dimension) {
//                return null;
//            }
//
//            @Override
//            public INDArray mean(INDArray result, int... dimension) {
//                return null;
//            }
//
//            @Override
//            public INDArray amean(int... dimension) {
//                return null;
//            }
//
//            @Override
//            public Number meanNumber() {
//                return null;
//            }
//
//            @Override
//            public Number ameanNumber() {
//                return null;
//            }
//
//            @Override
//            public INDArray var(int... dimension) {
//                return null;
//            }
//
//            @Override
//            public INDArray var(boolean biasCorrected, int... dimension) {
//                return null;
//            }
//
//            @Override
//            public Number varNumber() {
//                return null;
//            }
//
//            @Override
//            public INDArray max(int... dimension) {
//                return null;
//            }
//
//            @Override
//            public INDArray amax(int... dimension) {
//                return null;
//            }
//
//            @Override
//            public Number maxNumber() {
//                return null;
//            }
//
//            @Override
//            public Number amaxNumber() {
//                return null;
//            }
//
//            @Override
//            public INDArray min(int... dimension) {
//                return null;
//            }
//
//            @Override
//            public INDArray amin(int... dimension) {
//                return null;
//            }
//
//            @Override
//            public Number minNumber() {
//                return null;
//            }
//
//            @Override
//            public Number aminNumber() {
//                return null;
//            }
//
//            @Override
//            public INDArray sum(int... dimension) {
//                return null;
//            }
//
//            @Override
//            public Number scan(Condition condition) {
//                return null;
//            }
//
//            @Override
//            public INDArray sum(INDArray result, int... dimension) {
//                return null;
//            }
//
//            @Override
//            public Number sumNumber() {
//                return null;
//            }
//
//            @Override
//            public Number entropyNumber() {
//                return null;
//            }
//
//            @Override
//            public Number shannonEntropyNumber() {
//                return null;
//            }
//
//            @Override
//            public Number logEntropyNumber() {
//                return null;
//            }
//
//            @Override
//            public INDArray entropy(int... dimension) {
//                return null;
//            }
//
//            @Override
//            public INDArray shannonEntropy(int... dimension) {
//                return null;
//            }
//
//            @Override
//            public INDArray logEntropy(int... dimension) {
//                return null;
//            }
//
//            @Override
//            public void setStride(long... stride) {
//
//            }
//
//            @Override
//            public void setShape(long... shape) {
//
//            }
//
//            @Override
//            public void setShapeAndStride(int[] shape, int[] stride) {
//
//            }
//
//            @Override
//            public void setOrder(char order) {
//
//            }
//
//            @Override
//            public INDArray subArray(ShapeOffsetResolution resolution) {
//                return null;
//            }
//
//            @Override
//            public INDArray subArray(long[] offsets, int[] shape, int[] stride) {
//                return null;
//            }
//
//            @Override
//            public INDArray getScalar(int... indices) {
//                return null;
//            }
//
//            @Override
//            public INDArray getScalar(long... indices) {
//                return null;
//            }
//
//            @Override
//            public int getInt(int... indices) {
//                return 0;
//            }
//
//            @Override
//            public double getDouble(int... indices) {
//                return 0;
//            }
//
//            @Override
//            public double getDouble(long... indices) {
//                return 0;
//            }
//
//            @Override
//            public float getFloat(int[] indices) {
//                return 0;
//            }
//
//            @Override
//            public float getFloat(long[] indices) {
//                return 0;
//            }
//
//            @Override
//            public double getDouble(long i) {
//                return 0;
//            }
//
//            @Override
//            public double getDouble(long i, long j) {
//                return 0;
//            }
//
//            @Override
//            public float getFloat(long i) {
//                return 0;
//            }
//
//            @Override
//            public float getFloat(long i, long j) {
//                return 0;
//            }
//
//            @Override
//            public INDArray dup() {
//                return null;
//            }
//
//            @Override
//            public INDArray dup(char order) {
//                return null;
//            }
//
//            @Override
//            public INDArray ravel() {
//                return null;
//            }
//
//            @Override
//            public INDArray ravel(char order) {
//                return null;
//            }
//
//            @Override
//            public void setData(DataBuffer data) {
//
//            }
//
//            @Override
//            public long slices() {
//                return 0;
//            }
//
//            @Override
//            public int getTrailingOnes() {
//                return 0;
//            }
//
//            @Override
//            public int getLeadingOnes() {
//                return 0;
//            }
//
//            @Override
//            public INDArray slice(long i, int dimension) {
//                return null;
//            }
//
//            @Override
//            public INDArray slice(long i) {
//                return null;
//            }
//
//            @Override
//            public long offset() {
//                return 0;
//            }
//
//            @Override
//            public long originalOffset() {
//                return 0;
//            }
//
//            @Override
//            public INDArray reshape(char order, long... newShape) {
//                return null;
//            }
//
//            @Override
//            public INDArray reshape(char order, int... newShape) {
//                return null;
//            }
//
//            @Override
//            public INDArray reshape(char order, int rows, int columns) {
//                return null;
//            }
//
//            @Override
//            public INDArray reshape(long... newShape) {
//                return null;
//            }
//
//            @Override
//            public INDArray reshape(int[] shape) {
//                return null;
//            }
//
//            @Override
//            public INDArray reshape(long rows, long columns) {
//                return null;
//            }
//
//            @Override
//            public INDArray transpose() {
//                return null;
//            }
//
//            @Override
//            public INDArray transposei() {
//                return null;
//            }
//
//            @Override
//            public INDArray swapAxes(int dimension, int with) {
//                return null;
//            }
//
//            @Override
//            public INDArray permute(int... rearrange) {
//                return null;
//            }
//
//            @Override
//            public INDArray permutei(int... rearrange) {
//                return null;
//            }
//
//            @Override
//            public INDArray dimShuffle(Object[] rearrange, int[] newOrder, boolean[] broadCastable) {
//                return null;
//            }
//
//            @Override
//            public INDArray dimShuffle(Object[] rearrange, long[] newOrder, boolean[] broadCastable) {
//                return null;
//            }
//
//            @Override
//            public INDArray getColumn(long i) {
//                return null;
//            }
//
//            @Override
//            public INDArray getRow(long i) {
//                return null;
//            }
//
//            @Override
//            public int columns() {
//                return 0;
//            }
//
//            @Override
//            public int rows() {
//                return 0;
//            }
//
//            @Override
//            public boolean isColumnVector() {
//                return false;
//            }
//
//            @Override
//            public boolean isRowVector() {
//                return false;
//            }
//
//            @Override
//            public boolean isColumnVectorOrScalar() {
//                return false;
//            }
//
//            @Override
//            public boolean isRowVectorOrScalar() {
//                return false;
//            }
//
//            @Override
//            public boolean isVector() {
//                return false;
//            }
//
//            @Override
//            public boolean isVectorOrScalar() {
//                return false;
//            }
//
//            @Override
//            public boolean isSquare() {
//                return false;
//            }
//
//            @Override
//            public boolean isMatrix() {
//                return false;
//            }
//
//            @Override
//            public boolean isScalar() {
//                return false;
//            }
//
//            @Override
//            public long[] shape() {
//                return new long[0];
//            }
//
//            @Override
//            public long[] stride() {
//                return new long[0];
//            }
//
//            @Override
//            public char ordering() {
//                return 0;
//            }
//
//            @Override
//            public long size(int dimension) {
//                return 0;
//            }
//
//            @Override
//            public long length() {
//                return 0;
//            }
//
//            @Override
//            public long lengthLong() {
//                return 0;
//            }
//
//            @Override
//            public INDArray broadcast(long... shape) {
//                return null;
//            }
//
//            @Override
//            public INDArray broadcast(INDArray result) {
//                return null;
//            }
//
//            @Override
//            public Object element() {
//                return null;
//            }
//
//            @Override
//            public DataBuffer data() {
//                return null;
//            }
//
//            @Override
//            public boolean equalsWithEps(Object o, double eps) {
//                return false;
//            }
//
//            @Override
//            public boolean equalShapes(INDArray other) {
//                return false;
//            }
//
//            @Override
//            public INDArray unsafeDuplication() {
//                return null;
//            }
//
//            @Override
//            public INDArray unsafeDuplication(boolean blocking) {
//                return null;
//            }
//
//            @Override
//            public INDArray remainder(INDArray denominator) {
//                return null;
//            }
//
//            @Override
//            public INDArray remainder(INDArray denominator, INDArray result) {
//                return null;
//            }
//
//            @Override
//            public INDArray remainder(Number denominator) {
//                return null;
//            }
//
//            @Override
//            public INDArray remainder(Number denominator, INDArray result) {
//                return null;
//            }
//
//            @Override
//            public INDArray remainderi(INDArray denominator) {
//                return null;
//            }
//
//            @Override
//            public INDArray remainderi(Number denominator) {
//                return null;
//            }
//
//            @Override
//            public INDArray fmod(INDArray denominator) {
//                return null;
//            }
//
//            @Override
//            public INDArray fmod(INDArray denominator, INDArray result) {
//                return null;
//            }
//
//            @Override
//            public INDArray fmod(Number denominator) {
//                return null;
//            }
//
//            @Override
//            public INDArray fmod(Number denominator, INDArray result) {
//                return null;
//            }
//
//            @Override
//            public INDArray fmodi(INDArray denominator) {
//                return null;
//            }
//
//            @Override
//            public INDArray fmodi(Number denominator) {
//                return null;
//            }
//
//            @Override
//            public INDArray argMax(int... dimension) {
//                return null;
//            }
//
//            @Override
//            public boolean isAttached() {
//                return false;
//            }
//
//            @Override
//            public boolean isInScope() {
//                return false;
//            }
//
//            @Override
//            public INDArray detach() {
//                return null;
//            }
//
//            @Override
//            public INDArray leverage() {
//                return null;
//            }
//
//            @Override
//            public INDArray leverageTo(String id) {
//                return null;
//            }
//
//            @Override
//            public INDArray leverageTo(String id, boolean enforceExistence) throws Nd4jNoSuchWorkspaceException {
//                return null;
//            }
//
//            @Override
//            public INDArray leverageOrDetach(String id) {
//                return null;
//            }
//
//            @Override
//            public INDArray migrate() {
//                return null;
//            }
//
//            @Override
//            public INDArray migrate(boolean detachOnNoWs) {
//                return null;
//            }
//
//            @Override
//            public Number percentileNumber(Number percentile) {
//                return null;
//            }
//
//            @Override
//            public Number medianNumber() {
//                return null;
//            }
//
//            @Override
//            public INDArray median(int... dimension) {
//                return null;
//            }
//
//            @Override
//            public INDArray percentile(Number percentile, int... dimension) {
//                return null;
//            }
//
//            @Override
//            public DataBuffer getVectorCoordinates() {
//                return null;
//            }
//
//            @Override
//            public INDArray toDense() {
//                return null;
//            }
//
//            @Override
//            public int nnz() {
//                return 0;
//            }
//
//            @Override
//            public SparseFormat getFormat() {
//                return null;
//            }
//
//            @Override
//            public int[] flags() {
//                return new int[0];
//            }
//
//            @Override
//            public int[] hiddenDimensions() {
//                return new int[0];
//            }
//
//            @Override
//            public int[] sparseOffsets() {
//                return new int[0];
//            }
//
//            @Override
//            public int underlyingRank() {
//                return 0;
//            }
//
//            @Override
//            public int toFlatArray(FlatBufferBuilder builder) {
//                return 0;
//            }
//
//            @Override
//            public INDArray convertToHalfs() {
//                return null;
//            }
//
//            @Override
//            public INDArray convertToFloats() {
//                return null;
//            }
//
//            @Override
//            public INDArray convertToDoubles() {
//                return null;
//            }
//
//            @Override
//            public boolean isEmpty() {
//                return false;
//            }
//
//            @Override
//            public long[] shapeInfoJava() {
//                return new long[0];
//            }
//
//            @Override
//            public DataBuffer.Type dataType() {
//                return null;
//            }
//        });
        Tensor testTensor = new Tensor() {
            @Override
            public void forward() {

            }

            @Override
            public void backward() {

            }

            @Override
            public Constant[] parameters() {
                return new Constant[0];
            }
        };

        Linear linear = new Linear(1,1);
        SoftmaxWithLoss softmaxWithLoss = new SoftmaxWithLoss(testTensor);
        System.out.println(linear.toString());
//        Conv2d conv2d = new Conv2d(1, true);
//        MaxPool2d maxPool2d = new MaxPool2d(1);
//        MSELoss mseLoss = new MSELoss(testConstant);
//        ReLU reLU = new ReLU();
//        Sequential sequential = new Sequential();
//        SoftmaxWithLoss softmaxWithLoss = new SoftmaxWithLoss(testTensor);
//        WeightInit weightInit = new WeightInit();
        ObjectMapper objectMapper = new ObjectMapper();
//        objectMapper.enableDefaultTyping();
        SoftmaxWithLoss testSoftmax = null;
        try {
            objectMapper.writeValue(new File(".\\softmaxWithLoss.json"), softmaxWithLoss);
            String temp = objectMapper.writeValueAsString(softmaxWithLoss);
            System.out.println(temp);
            testSoftmax = (SoftmaxWithLoss) objectMapper.readValue(temp, Tensor.class);
        } catch (IOException e) {
            e.printStackTrace();
        }


        assert testSoftmax != null;
        System.out.println(testSoftmax.toString());

//        Conv2d testConv2d;
//        MaxPool2d testMaxPool2d;
//        MSELoss testMSELoss;
//        ReLU testReLU;
//        Sequential testSequential;
//        SoftmaxWithLoss testSoftmaxWithLoss;
//        WeightInit testWeightInit;

//        Serialization.save(linear, ".\\resource\\Serialization\\linear.json", "json");
//        testLinear = (Linear) Serialization.load(".\\resource\\Serialization\\linear.json");
//        Serialization.save(conv2d, ".\\resource\\Serialization\\conv2d.json", "json");
//        testConv2d = (Conv2d) Serialization.load(".\\resource\\Serialization\\conv2d.json");
//        Serialization.save(maxPool2d, ".\\resource\\Serialization\\maxPool2d.json", "json");
//        testMaxPool2d = (MaxPool2d) Serialization.load(".\\resource\\Serialization\\maxPool2d.json");
//        Serialization.save(mseLoss, ".\\resource\\Serialization\\mseLoss.json", "json");
//        testMSELoss = (MSELoss) Serialization.load(".\\resource\\Serialization\\mseLoss.json");
//        Serialization.save(reLU, ".\\resource\\Serialization\\reLU.json", "json");
//        testReLU = (ReLU) Serialization.load(".\\resource\\Serialization\\reLU.json");
//        Serialization.save(sequential, ".\\resource\\Serialization\\sequential.json", "json");
//        testSequential = (Sequential) Serialization.load(".\\resource\\Serialization\\sequential.json");
//        Serialization.save(softmaxWithLoss, ".\\resource\\Serialization\\softmaxWithLoss.json", "json");
//        testSoftmaxWithLoss = (SoftmaxWithLoss) Serialization.load(".\\resource\\Serialization\\softmaxWithLoss.json");
//        Serialization.save(weightInit, ".\\resource\\Serialization\\weightInit.json", "json");
//        testWeightInit = (WeightInit) Serialization.load(".\\resource\\Serialization\\weightInit.json");

//        assert testLinear != null;
//        System.out.println(testLinear.toString());
//        System.out.println(testLinear.out);
//        System.out.println(testLinear.dout);
//
//        assert testConv2d != null;
//        System.out.println(testConv2d.toString());
//        System.out.println(testConv2d.out);
//        System.out.println(testConv2d.dout);
//
//        assert testMaxPool2d != null;
//        System.out.println(testMaxPool2d.toString());
//        System.out.println(testMaxPool2d.out);
//        System.out.println(testMaxPool2d.dout);
//
//        assert testMSELoss != null;
//        System.out.println(testMSELoss.toString());
//        System.out.println(testMSELoss.out);
//        System.out.println(testMSELoss.dout);
//
//        assert testReLU != null;
//        System.out.println(testReLU.toString());
//        System.out.println(testReLU.out);
//        System.out.println(testReLU.dout);
//
//        assert testSequential != null;
//        System.out.println(testSequential.toString());
//        System.out.println(testSequential.out);
//        System.out.println(testSequential.dout);
//
//        assert testSoftmaxWithLoss != null;
//        System.out.println(testSoftmaxWithLoss.toString());
//        System.out.println(testSoftmaxWithLoss.out);
//        System.out.println(testSoftmaxWithLoss.dout);
//
//        assert testWeightInit != null;
//        System.out.println(testWeightInit.toString());
//        System.out.println(testWeightInit.out);
//        System.out.println(testWeightInit.dout);


    }

}
