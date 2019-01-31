package com.jstarcraft.module.datatran;

import com.jstarcraft.module.datatran.labelmaker.FuncLabelMaker;
import com.jstarcraft.module.datatran.transform.Transform;
import com.jstarcraft.module.datatran.transform.integrate.BranchTransform;
import com.jstarcraft.module.datatran.transform.integrate.FMIImageTransform;
import com.jstarcraft.module.datatran.transform.nd4j.CookDSTransform;
import com.jstarcraft.module.datatran.transform.rawdataset.*;
import com.jstarcraft.module.datatran.transform.pretreatment.FlatRootTransform;
import com.jstarcraft.module.datatran.transform.pretreatment.IndexTransform;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

public class DataTranTest {

    @Before
    public void before() {

    }

    @Test
    public void testType() {

    }

    @Test
    public void testTran() {
        String pathname = "F:/Programs/moduleimage/ImagePipeline/label3";
        System.out.println(pathname);

        String[] allowedExtensions = {"tif", "jpg", "png", "jpeg", "bmp", "JPEG", "JPG", "TIF", "PNG"};

        Transform t1 = new FlatRootTransform(allowedExtensions);
        Transform t2 = new MakeLabelTransform<>(new FuncLabelMaker<>((File f)->f.getParentFile().getName()));
        Transform t3 = new BalanceLabelTransform<>();
        Transform t4 = new ShuffleRDSTransform<>();

        List<String> index2Label = new ArrayList<>();

        BranchTransform<File, String, INDArray, INDArray> t5 = new BranchTransform<>();
        t5.addFTransform(new FMIImageTransform());
        t5.addLTransform(new IndexTransform(index2Label));

        Transform t6 = new CookDSTransform();

        DataDriver<File,DataSet> dataDriver
                = new DataDriver<>(new File(pathname), t1, t2, t3, t4, t5, t6);
        try {
            DataSet translation = dataDriver.translation();
            System.out.println(translation);
            System.out.println(index2Label.toString());
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @After
    public void after() {

    }
}
