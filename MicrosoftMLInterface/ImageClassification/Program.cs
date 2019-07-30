using ImageClassification.ModelScorer;
using System.IO;


namespace ImageClassification
{
    public class Program
    {
        private static readonly string assetsRelativePath = @"../../../assets";
        private static readonly string assetsPath = GetAbsolutePath(assetsRelativePath);

        private static readonly string tagsTsv = @"D:\Datasets\CatsDogs\image_list.tsv";
        private static readonly string imagesFolder = @"D:\Datasets\CatsDogs";
        private static readonly string labelsTxt = Path.Combine(assetsPath, "inputs", "catsdogs", "labels.txt");
        private static readonly string pathToModel = Path.Combine(assetsPath, "inputs", "catsdogsNet");

        

        static void Main(string[] args)
        {
            var modelScorer = new TFModelScorer(pathToModel, "lambda_input", "dense/Sigmoid");
            modelScorer.PredictDataUsingModel(tagsTsv, imagesFolder, labelsTxt);

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
