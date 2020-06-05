using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DNN
{
   class  Dataset
    {
        private List<double[]> InputDataset;
        private List<double[]> OutputDataset;
        private int InputLength;
        private int OutputLength;

        public Dataset(int input_length,int output_length)
        {
            InputLength = input_length;
            OutputLength = output_length;

            InputDataset = new List<double[]>();
            OutputDataset = new List<double[]>();
        }
        public void LoadDataset(string train)
        {
            using (var reader = new System.IO.StreamReader(train))
            {

                while (!reader.EndOfStream)
                {
                    var line = reader.ReadLine();//read string line 
                    var values = line.Split(',').Select(x=> double.Parse(x));//store values and convert them to double 

                    InputDataset.Add(values.Take(InputLength).ToArray());//store the input data array to the this list
                    OutputDataset.Add(values.Skip(InputLength).ToArray());//store the output data array to the this list
                }
            }
        }
    }
}
