package com.jstarcraft.module.math.structure.matrix;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class NdMatrix extends DefaultMatrix {
    private INDArray array;
    private static INDArray opMatrix;

    public NdMatrix(INDArray array){
        this.array = array;
    }

    public NdMatrix(double[] array){
        this.array = Nd4j.create(array);
    }

    public NdMatrix(double[] array, int[] shape){
        this.array = Nd4j.create(array, shape);
    }

    public static NdMatrix rand(int[] shape){
        return new NdMatrix(Nd4j.rand(shape));
    }

    public static NdMatrix rand(int r, int c){
        return new NdMatrix(Nd4j.rand(r,c));
    }

    @Override
    public Number getEle(int rowIndex, int colIndex) {
        return array.getDouble(rowIndex, colIndex);
    }

    @Override
    public void setEle(int rowIndex, int colIndex, Number value) {
        array.put(rowIndex, colIndex, value);
    }

    @Override
    public int nRow() {
        return array.rows();
    }

    @Override
    public int nCol() {
        return array.columns();
    }

    @Override
    public Matrix create(int nRow, int nCol) {
        return new NdMatrix(Nd4j.zeros(nRow, nCol));
    }

    @Override
    public Matrix mul(Matrix m) {
        genOp(m);
        return new NdMatrix(array.mul(opMatrix));
    }

    @Override
    public Matrix muli(Matrix m) {
        genOp(m);
        array.muli(opMatrix);
        return this;
    }

    @Override
    public Matrix add(Matrix m) {
        genOp(m);
        return new NdMatrix(array.add(opMatrix));
    }

    @Override
    public Matrix addi(Matrix m) {
        genOp(m);
        array.addi(opMatrix);
        return this;
    }

    @Override
    public Matrix subi(Matrix m) {
        genOp(m);
        array.subi(opMatrix);
        return this;
    }

    @Override
    public Matrix sub(Matrix m) {
        genOp(m);
        return new NdMatrix(array.sub(opMatrix));
    }

    @Override
    public Matrix div(Number n) {
        return new NdMatrix(array.div(n));
    }

    @Override
    public Matrix divi(Number n) {
        array.divi(n);
        return this;
    }

    private static void genOp(Matrix m){
        if(m instanceof NdMatrix){
            opMatrix = NdMatrix.class.cast(m).array;
        }else{
            throw new UnsupportedOperationException();
        }
    }

    @Override
    public String toString() {
        return array.toString();
    }

    public INDArray getArray() {
        return array;
    }
}
