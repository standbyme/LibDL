package com.jstarcraft.module.math.structure.matrix;

public interface Matrix {

    Number getEle(int rowIndex, int colIndex);

    void setEle(int rowIndex, int colIndex, Number value);

    int nRow();

    int nCol();

    Matrix create(int nRow, int nCol);

    Matrix addi(Matrix m);

    Matrix add(Matrix m);

    Matrix mul(Matrix m);

    Matrix muli(Matrix m);

    Matrix sub(Matrix m);

    Matrix subi(Matrix m);

    Matrix dot(Matrix m);

    Matrix doti(Matrix m);

    Matrix mul(Number n);

    Matrix muli(Number n);

    Matrix transpose();

    Matrix div(Number n);

    Matrix divi(Number n);

    Matrix transposei();

    void forEach(MatrixIterator matrixIterator);
}
