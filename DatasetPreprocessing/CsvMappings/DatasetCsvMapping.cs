using DatasetPreprocessing.Models;
using TinyCsvParser.Mapping;

namespace DatasetPreprocessing.CsvMappings
{
    public class DatasetCsvMapping : CsvMapping<DataModel>
    {
        public DatasetCsvMapping() : base()
        {
            int i = 0;
            MapProperty(i++, x => x.DivisionNr);
            MapProperty(i++, x => x.DateOfReport);
            MapProperty(i++, x => x.DateOfOccurrence);
            MapProperty(i++, x => x.Time);
            MapProperty(i++, x => x.Area);
            MapProperty(i++, x => x.AreaName);
            MapProperty(i++, x => x.SubAreaCode);
            MapProperty(i++, x => x.Part12);
            MapProperty(i++, x => x.CrimeCode);
            MapProperty(i++, x => x.CrimeCommited);
            MapProperty(i++, x => x.ModusOperandi);
            MapProperty(i++, x => x.VictAge);
            MapProperty(i++, x => x.VictSex);
            MapProperty(i++, x => x.VictDescent);
            MapProperty(i++, x => x.PremisCd);
            MapProperty(i++, x => x.PremisDecs);
            MapProperty(i++, x => x.WeaponUsedCd);
            MapProperty(i++, x => x.WeaponDesc);
            MapProperty(i++, x => x.Status);
            MapProperty(i++, x => x.StatusDesc);
            MapProperty(i++, x => x.CrmCd1);
            MapProperty(i++, x => x.CrmCd2);
            MapProperty(i++, x => x.CrmCd3);
            MapProperty(i++, x => x.CrmCd4);
            MapProperty(i++, x => x.Location);
            MapProperty(i++, x => x.CrossStreet);
            MapProperty(i++, x => x.Latitude);
            MapProperty(i++, x => x.Longitude);
        }
    }
}