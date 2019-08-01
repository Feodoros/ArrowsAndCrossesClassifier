using ImageClassification.ModelScorer;
using Microsoft.ML.Data;

namespace ImageClassification.ImageDataStructures
{
    public class ImageNetPrediction
    {
        [ColumnName(TFModelScorer.output)] 
        public float[] PredictedLabels;
    }
}
