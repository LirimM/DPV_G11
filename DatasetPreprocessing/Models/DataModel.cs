namespace DatasetPreprocessing.Models
{
    public class DataModel
    {
        public string DivisionNr { get; set; } = null!;
        public DateTime DateOfReport { get; set; }
        public DateTime DateOfOccurrence { get; set; }
        public int Time { get; set; }
        public string Area { get; set; } = null!;
        public string AreaName { get; set; } = null!;
        public int SubAreaCode { get; set; }
        public int Part12 { get; set; }
        public int CrimeCode { get; set; }
        public string CrimeCommited { get; set; } = null!;
        public string ModusOperandi { get; set; } = null!;
        public string VictAge { get; set; } = null!;
        public string VictSex { get; set; } = null!;
        public string VictDescent { get; set; } = null!;
        public int PremisCd { get; set; }
        public string PremisDecs { get; set; } = null!;
        public string WeaponUsedCd { get; set; } = null!;
        public string WeaponDesc { get; set; } = null!;
        public string Status { get; set; } = null!;
        public string StatusDesc { get; set; } = null!;
        public string CrmCd1 { get; set; } = null!;
        public string CrmCd2 { get; set; } = null!;
        public string CrmCd3 { get; set; } = null!;
        public string CrmCd4 { get; set; } = null!;
        public string Location { get; set; } = null!;
        public string CrossStreet { get; set; } = null!;
        public decimal Latitude { get; set; }
        public decimal Longitude { get; set; }

    }
}
