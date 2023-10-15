package MainClasses.EliminateInputParameterSelection;

/**
 * Created by Hajar Rahimi
 */
public class Result {
//--------------- Instance Variable ----------------
    double Accuracy;
    protected double Precision;
    private double Recall;
    private double F1Score;
//--------------- Constructors ----------------
    public Result(double accuracy, double precision, double recall, double f1Score) {
        Accuracy = accuracy;
        Precision = precision;
        Recall = recall;
        F1Score = f1Score;
    }
//--------------- Methods ----------------
    public double getAccuracy() {
        return Accuracy;
    }

    public double getPrecision() {
        return Precision;
    }

    public double getRecall() {
        return Recall;
    }

    public double getF1Score() {
        return F1Score;
    }

}
