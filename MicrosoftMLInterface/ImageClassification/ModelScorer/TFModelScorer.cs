using System;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using Microsoft.ML;
using ImageClassification.ImageDataStructures;
using static ImageClassification.ModelScorer.ConsoleHelpers;
using static ImageClassification.ModelScorer.ModelHelpers;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Image;

namespace ImageClassification.ModelScorer
{
    public class TFModelScorer
    {
        private readonly string dataLocation;
        private readonly string imagesFolder;
        private readonly string modelLocation;
        private readonly string labelsLocation;
        private readonly MLContext mlContext;
        private static string ImageReal = nameof(ImageReal);

        public object ImagePixelExtractorTransform { get; private set; }

        public TFModelScorer(string dataLocation, string imagesFolder, string modelLocation, string labelsLocation)
        {
            this.dataLocation = dataLocation;
            this.imagesFolder = imagesFolder;
            this.modelLocation = modelLocation;
            this.labelsLocation = labelsLocation;
            mlContext = new MLContext();
        }

        public struct ImageNetSettings
        {
            public const int imageHeight = 150;
            public const int imageWidth = 150;
            public const float mean = 3;       
            public const bool channelsLast = true;
            public const float scale = 1 / 255f;
        }

        public struct InceptionSettings
        {
            // for checking tensor names, you can use tools like Netron,
            // which is installed by Visual Studio AI Tools

            // input tensor name
            public const string input = "lambda_input";

            // output tensor name
            public const string output = "dense/Sigmoid";

        }

        public void Score()
        {
            var model = LoadModel(dataLocation, imagesFolder, modelLocation);

            var predictions = PredictDataUsingModel(dataLocation, imagesFolder, labelsLocation, model).ToArray();

        }

        private PredictionEngine<ImageNetData, ImageNetPrediction> LoadModel(string dataLocation, string imagesFolder, string modelLocation)
        {
            ConsoleWriteHeader("Read model");
            Console.WriteLine($"Model location: {modelLocation}");
            Console.WriteLine($"Images folder: {imagesFolder}");
            Console.WriteLine($"Training file: {dataLocation}");
            Console.WriteLine($"Default parameters: image size=({ImageNetSettings.imageWidth},{ImageNetSettings.imageHeight}), image mean: {ImageNetSettings.mean}");

            var data = mlContext.Data.LoadFromTextFile<ImageNetData>(dataLocation, hasHeader: true);

            // mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: InceptionSettings.input, inputColumnName: nameof(ImageNetData.Label))

            var pipeline = mlContext.Transforms.LoadImages(outputColumnName: InceptionSettings.input, imageFolder: imagesFolder, inputColumnName: nameof(ImageNetData.ImagePath))
                .Append(mlContext.Transforms.ResizeImages(outputColumnName: InceptionSettings.input, imageWidth: ImageNetSettings.imageWidth, imageHeight: ImageNetSettings.imageHeight, inputColumnName: InceptionSettings.input))
                .Append(mlContext.Transforms.ExtractPixels(outputColumnName: InceptionSettings.input, interleavePixelColors: ImageNetSettings.channelsLast, offsetImage: ImageNetSettings.mean, scaleImage: ImageNetSettings.scale))
                .Append(mlContext.Model.LoadTensorFlowModel(modelLocation)
                    .ScoreTensorFlowModel(outputColumnNames: new[] { InceptionSettings.output },
                                          inputColumnNames: new[] { InceptionSettings.input }, addBatchDimensionInput: false));        
                        
            ITransformer model = pipeline.Fit(data);

            var predictionEngine = mlContext.Model.CreatePredictionEngine<ImageNetData, ImageNetPrediction>(model);           

            // mlContext.Model.Save(model, data.Schema, "model.zip");

            return predictionEngine;
        }

        protected IEnumerable<ImageNetData> PredictDataUsingModel(string testLocation, 
                                                                  string imagesFolder, 
                                                                  string labelsLocation, 
                                                                  PredictionEngine<ImageNetData, ImageNetPrediction> model)
        {
            ConsoleWriteHeader("Classificate images");
            Console.WriteLine($"Images folder: {imagesFolder}");
            Console.WriteLine($"Training file: {testLocation}");
            Console.WriteLine($"Labels file: {labelsLocation}");

            double acc = 0;
            double tp = 0;
            double fp = 0;
            double fn = 0;

            var labels = ModelHelpers.ReadLabels(labelsLocation);

            var testData = ImageNetData.ReadFromCsv(testLocation, imagesFolder);
            var count = 0;

            Console.ForegroundColor = ConsoleColor.Red;

            foreach (var sample in testData)
            {
                var probs = model.Predict(sample).PredictedLabels;
                var imageData = new ImageNetDataProbability()
                {
                    ImagePath = sample.ImagePath,
                    Label = sample.Label
                };
                (imageData.PredictedLabel, imageData.Probability) = GetBestLabel(labels, probs);
                 
                //imageData.ConsoleWrite();

                acc += imageData.Label == imageData.PredictedLabel ? 1 : 0;
                tp += imageData.PredictedLabel == imageData.Label && imageData.Label == labels[1] ? 1 : 0;
                fp += imageData.Label == labels[0] && imageData.PredictedLabel == labels[1] ? 1 : 0;
                fn += imageData.Label == labels[1] && imageData.PredictedLabel == labels[0] ? 1 : 0;

                count++;
                if (count % 100 == 0)
                {
                    Console.WriteLine($"{count} images have been processed");
                }

                yield return imageData;
            }

            acc = (acc / testData.ToList().Count()) * 100;
            double prec = tp / (tp + fp);
            double rec = tp / (tp + fn);
            double f1 = 2 * (prec * rec) / (prec + rec);

           
            Console.WriteLine($"Accuracy: {acc}%, precision: {prec}, recall: {rec}, f1 score: {f1}");
        }
    }
}
