package LibDL.nn;

public class AlphaDropout extends Dropout {
    public AlphaDropout(double p, boolean train) {
        super(p, train, false, false);
    }

    protected AlphaDropout(double p, boolean t, boolean f, boolean a) {
        super(p, t, f, a);
    }
}
