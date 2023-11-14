using DatasetPreprocessing.CsvMappings;
using DatasetPreprocessing.Models;
using System.IO.Compression;
using System.Text;
using TinyCsvParser;
using TinyCsvParser.Mapping;

namespace DatasetPreprocessing
{
    public class Preprocessing
    {
        static void Main(string[] args)
        {
            Console.WriteLine("asd");

            string solutionDirectory = Directory.GetParent(Directory.GetCurrentDirectory())!.Parent!.Parent!.Parent!.FullName;

            string path = Path.GetFullPath(Path.Combine(solutionDirectory, "Crime_Data_from_2020_to_Present_20231113.zip"));

            byte[] zipBytes = File.ReadAllBytes(path);

            using (MemoryStream zipStream = new MemoryStream(zipBytes))
            {
                using (ZipArchive archive = new ZipArchive(zipStream, ZipArchiveMode.Read))
                {
                    byte[][] extractedFileBytes = new byte[archive.Entries.Count][];

                    ZipArchiveEntry entry = archive.Entries[1];

                    using (Stream entryStream = entry.Open())
                    {
                        CsvParserOptions csvParserOptions = new CsvParserOptions(true, ',');
                        CsvParser<DataModel> csvParser = new CsvParser<DataModel>(csvParserOptions, new DatasetCsvMapping());
                        ParallelQuery<CsvMappingResult<DataModel>> records = csvParser.ReadFromStream(entryStream, Encoding.UTF8);

                        foreach (CsvMappingResult<DataModel>? record in records.Take(10).ToList())
                        {
                            if (record.IsValid)
                            {
                                Console.WriteLine(record.Result.DivisionNr);
                            }
                        }

                    }
                }
            }




        }
    }
}
