package com.jstarcraft.module.math.structure.matrix;

import org.nd4j.linalg.api.ndarray.INDArray;

public abstract class DefaultMatrix implements Matrix {

    @Override
    public Matrix addi(Matrix m) {
        m.forEach((n, row, col) -> setEle(row, col, n.doubleValue() + getEle(row, col).doubleValue()));
        return this;
    }

    @Override
    public Matrix add(Matrix m) {
        Matrix result = create(nRow(),nCol());
        m.forEach((n, row, col) -> result.setEle(row, col, n.doubleValue() + getEle(row, col).doubleValue()));
        return result;
    }

    @Override
    public Matrix dot(Matrix m) {
        Matrix result = create(nRow(),nCol());
        m.forEach((n, row, col) -> result.setEle(row, col, n.doubleValue() * getEle(row, col).doubleValue()));
        return result;
    }

    @Override
    public Matrix doti(Matrix m) {
        m.forEach((n, row, col) -> setEle(row, col, n.doubleValue() * getEle(row, col).doubleValue()));
        return this;
    }

    @Override
    public Matrix sub(Matrix m) {
        Matrix result = create(nRow(),nCol());
        m.forEach((n, row, col) -> result.setEle(row, col, n.doubleValue() - getEle(row, col).doubleValue()));
        return result;
    }

    @Override
    public Matrix subi(Matrix m) {
        m.forEach((n, row, col) -> setEle(row, col, n.doubleValue() - getEle(row, col).doubleValue()));
        return this;
    }

    @Override
    public Matrix mul(Matrix m) {
        return null;
    }

    @Override
    public Matrix muli(Matrix m) {
        return null;
    }

    @Override
    public Matrix mul(Number n) {
        return null;
    }

    @Override
    public Matrix muli(Number n) {
        return null;
    }

    @Override
    public Matrix transpose() {
        return null;
    }

    @Override
    public Matrix div(Number n) {
        return null;
    }

    @Override
    public Matrix divi(Number n) {
        return null;
    }

    @Override
    public Matrix transposei() {
        return null;
    }

    @Override
    public void forEach(MatrixIterator matrixIterator) {
        for(int i = 0; i < nCol();i++){
            for(int j = 0;j < nRow(); j++){
                matrixIterator.accept(getEle(i,j),i ,j);
            }
        }
    }

}
