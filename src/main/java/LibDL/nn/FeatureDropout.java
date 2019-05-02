package LibDL.nn;

public class FeatureDropout extends Dropout {
    public FeatureDropout(double p, boolean train) {
        super(p, train, false, false);
    }

    protected FeatureDropout(double p, boolean t, boolean f, boolean a) {
        super(p, t, f, a);
    }
}
