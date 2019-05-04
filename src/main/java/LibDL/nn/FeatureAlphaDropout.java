package LibDL.nn;

public class FeatureAlphaDropout extends AlphaDropout {
    public FeatureAlphaDropout(double p, boolean train) {
        super(p, train, true, true);
    }
}
