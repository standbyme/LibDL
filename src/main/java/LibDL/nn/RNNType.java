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

    public String toString(){
        switch (value){
            case 1:return "RNN";
            case 3:return "GRU";
            case 4:return "LSTM";
        }
        return "RNN";
    }
}