using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using ImageClassification.ImageDataStructures;
using static ImageClassification.ModelScorer.ConsoleHelpers;
using static ImageClassification.ModelScorer.ModelHelpers;

namespace ImageClassification.ModelScorer
{
    public class TFModelScorer
    {
        private readonly MLContext mlctx;
        private readonly ITransformer model;

        // Default parameters
        private int imageHeight = 160;
        private int imageWidth = 160;
        private float mean = 3;
        private bool channelsLast = true;
        private float scale = 1 / 255f;

        public const string input  = "lambda_input";
        public const string output = "dense/Sigmoid";

        public TFModelScorer(string modelLocation)
        {
            mlctx = new MLContext();
            model = LoadModel(modelLocation);
        }

        private ITransformer LoadModel(string modelLocation)
        {
            ConsoleWriteHeader("Read model");
            Console.WriteLine($"Model location: {modelLocation}");
            Console.WriteLine($"Parameters: image size=({imageWidth},{imageHeight}), image mean: {mean}");

            // A way to avoid undesirable data validation
            var placeholder = mlctx.Data.CreateTextLoader<ImageNetData>().Load("");
            var pathholder = "";

            var pipeline = mlctx.Transforms.LoadImages(outputColumnName: input, imageFolder: pathholder, inputColumnName: nameof(ImageNetData.ImagePath))
                .Append(mlctx.Transforms.ResizeImages(outputColumnName: input, imageWidth: imageWidth, imageHeight: imageHeight, inputColumnName: input))
                .Append(mlctx.Transforms.ExtractPixels(outputColumnName: input, interleavePixelColors: channelsLast, offsetImage: mean, scaleImage: scale))
                .Append(mlctx.Model.LoadTensorFlowModel(modelLocation)
                    .ScoreTensorFlowModel(outputColumnNames: new[] { output },
                                          inputColumnNames:  new[] { input  }, addBatchDimensionInput: false));        
                       
            ITransformer model = pipeline.Fit(placeholder);

            return model;
        }

        public void SetImageParameters(int height, int width, float mean, float scale, bool channelsLast)
        {
            this.imageHeight = height;
            this.imageWidth = width;
            this.mean = mean;
            this.scale = scale;
            this.channelsLast = channelsLast;
        }

        public void SaveModel(string path)
        {
            var schema = mlctx.Data.CreateTextLoader<ImageNetData>().Load("").Schema;
            mlctx.Model.Save(model, schema, Path.Combine(path, "model.zip"));
        }

        public void PredictDataUsingModel(string testLocation, string imagesFolder, string labelsLocation)
        {
            
            var watch = System.Diagnostics.Stopwatch.StartNew();

            ConsoleWriteHeader("Classify images");
            Console.WriteLine($"Images folder: {imagesFolder}");
            Console.WriteLine($"Training file: {testLocation}");
            Console.WriteLine($"Labels file: {labelsLocation}");

            double acc = 0;
            double tp  = 0;
            double fp  = 0;
            double fn  = 0;

            var labels = ReadLabels(labelsLocation);

            var testData = ImageNetData.ReadFromCsv(testLocation, imagesFolder);
            var dataLength = testData.ToList().Count();

            var count = 1;

            Console.ForegroundColor = ConsoleColor.Red;

            foreach (var sample in testData)
            {
                var engine = mlctx.Model.CreatePredictionEngine<ImageNetData, ImageNetPrediction>(model);

                var probs = engine.Predict(sample).PredictedLabels;
                var imageData = new ImageNetDataProbability()
                {
                    ImagePath = sample.ImagePath,
                    Label = sample.Label
                };
                (imageData.PredictedLabel, imageData.Probability) = GetBestLabel(labels, probs);
                 
                //imageData.ConsoleWrite();

                acc += (imageData.Label == imageData.PredictedLabel) ? 1 : 0;
                tp  += (imageData.PredictedLabel == imageData.Label) && (imageData.Label == labels[1]) ? 1 : 0;
                fp  += (imageData.Label == labels[0]) && (imageData.PredictedLabel == labels[1]) ? 1 : 0;
                fn  += (imageData.Label == labels[1]) && (imageData.PredictedLabel == labels[0]) ? 1 : 0;

                if (++count % 1000 == 0)
                {
                    Console.WriteLine($"{count} images have been processed");
                }
            }

            watch.Stop();

            acc = (acc / dataLength) * 100;
            double prec = tp / (tp + fp);
            double rec = tp / (tp + fn);
            double f1 = 2 * (prec * rec) / (prec + rec);

            var measuredTime = watch.ElapsedMilliseconds / 1000;

            Console.WriteLine($"Made predictions for {dataLength} images in {measuredTime} seconds. Accuracy: {acc}%, precision: {prec}, recall: {rec}, f1 score: {f1}");
        }
    }
}
