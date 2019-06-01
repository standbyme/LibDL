package LibDL.nn;

public enum RNNType {
    TYPE_RNN(1),
    TYPE_LSTM(4),
    TYPE_GRU(3);

    private final int value;

    RNNType(int value) {
        this.value = value;
    }

    public int gateSize() {
        return value;
    }
}