using ImageClassification.ModelScorer;
using System;
using System.IO;
using Microsoft.ML;
using ImageClassification.ImageDataStructures;


namespace ImageClassification
{
    public class Program
    {
        static void Main(string[] args)
        {
            string assetsRelativePath = @"../../../assets";
            string assetsPath = GetAbsolutePath(assetsRelativePath);

            var tagsTsv = Path.Combine(assetsPath, "inputs", "catsdogs", "image_list.tsv");
            var imagesFolder = Path.Combine(assetsPath, "inputs", "catsdogs", "images");  
            var labelsTxt = Path.Combine(assetsPath, "inputs", "catsdogs", "labels.txt");
            var pathToModel = @"D:\projects\Python projects\ArrowsAndCrosses\BinaryImageClassifier\models\zakladkiNetV2\1563973726";

            try
            {
                var modelScorer = new TFModelScorer(tagsTsv, imagesFolder, pathToModel, labelsTxt);
                modelScorer.Score();

            }
            catch (Exception ex)
            {
                ConsoleHelpers.ConsoleWriteException(ex.ToString());
            }

            ConsoleHelpers.ConsolePressAnyKey();
        }

        public static string GetAbsolutePath(string relativePath)
        {
            FileInfo _dataRoot = new FileInfo(typeof(Program).Assembly.Location);
            string assemblyFolderPath = _dataRoot.Directory.FullName;
            string fullPath = Path.Combine(assemblyFolderPath, relativePath);
            return fullPath;
        }
    }
}
