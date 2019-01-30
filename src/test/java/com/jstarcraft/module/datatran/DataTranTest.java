package com.jstarcraft.module.datatran;

import com.jstarcraft.module.datatran.entity.DataNode;
import com.jstarcraft.module.datatran.entity.RawDataSet;
import com.jstarcraft.module.datatran.labelmaker.DefaultPPLabelMaker;
import com.jstarcraft.module.datatran.transform.Image.ImageLoadTransform;
import com.jstarcraft.module.datatran.transform.Image.ImageND4JTransform;
import com.jstarcraft.module.datatran.transform.Transform;
import com.jstarcraft.module.datatran.transform.rawdataset.*;
import com.jstarcraft.module.datatran.transform.pretreatment.FlatRootTransform;
import com.jstarcraft.module.datatran.transform.pretreatment.IndexTransform;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.dataset.DataSet;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

public class DataTranTest {

    @Before
    public void before() {

    }

    @Test
    public void testTran() {
        String pathname = "F:/Programs/moduleimage/ImagePipeline";
        System.out.println(pathname);

        String[] allowedExtensions = {"tif", "jpg", "png", "jpeg", "bmp", "JPEG", "JPG", "TIF", "PNG"};

        Transform t1 = new FlatRootTransform(allowedExtensions);
        RDSTransform<File, String> t2 = new MakeLabelTransform<>(new DefaultPPLabelMaker());
        RDSTransform<File, String> t3 = new BalanceLabelTransform<>();
        RDSTransform<File, Integer> t4 = new ShuffleRDSTransform<>();

        List<String> index2Label = new ArrayList<>();

        BranchTransform<File, Integer> t5 = new BranchTransform<>();
        t5.addFTransform(new ImageLoadTransform());
        t5.addLTransform(new IndexTransform(index2Label));

        ImageND4JTransform t6 = new ImageND4JTransform();

        DataNode<DataSet> dataNode
                = new DataNode<>(new File(pathname), t1, t2, t3, t4, t5, t6);
        try {
            DataSet translation = dataNode.translation();
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
