using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DNN
{
   class  Dataset
    {
        public List<double[]> TrainingLable { get; set; }
        public List<double[]> TrainingInput{ get; set; }

        public double[][] TrainingLableArray { get; set; }
        public double[][] TrainingInputArray { get; set; }

        public List<double[]> TestingLable { get; set; }
        public List<double[]> TestingInput { get; set; }

        private int InputLength;
        private int LabletLength;

        public int TrainingLength { get { return TrainingLable.Count; } }//input dataset have the same length as output dataset
        public int TestingLength { get { return TestingLable.Count; } }//input dataset have the same length as output dataset

        public Dataset(int input_length,int Lable_length)
        {
            InputLength = input_length;
            LabletLength = Lable_length;

            TrainingInput = new List<double[]>();
            TrainingLable = new List<double[]>();

            TestingInput = new List<double[]>();
            TestingLable = new List<double[]>();

        }
        public void LoadTraining(string train)
        {
            using (var reader = new System.IO.StreamReader(train))
            {

                while (!reader.EndOfStream)
                {
                    var line = reader.ReadLine();//read string line 
                    var values = line.Split(',').Select(x=> double.Parse(x));//store values and convert them to double 

                    TrainingLable.Add(values.Take(LabletLength).ToArray());//store the input data array to the this list
                    TrainingInput.Add(values.Skip(LabletLength).ToArray());//store the output data array to the this list
                }
            }
            for (int i = 0; i < TrainingInput.Count; i++)
            {
                for (int j = 0; j < InputLength; j++)
                {
                    TrainingInput[i][j] /= 255;
                }
            }

            TrainingLableArray = TrainingLable.ToArray();
            TrainingInputArray = TrainingInput.ToArray();
        }
        public void LoadTesting(string train)
        {
            using (var reader = new System.IO.StreamReader(train))
            {

                while (!reader.EndOfStream)
                {
                    var line = reader.ReadLine();//read string line 
                    var values = line.Split(',').Select(x => double.Parse(x));//store values and convert them to double 

                    TestingLable.Add(values.Take(LabletLength).ToArray());//store the input data array to the this list
                    TestingInput.Add(values.Skip(LabletLength).ToArray());//store the output data array to the this list
                }
            }
            for (int i = 0; i < TestingInput.Count; i++)
            {
                for (int j = 0; j < InputLength; j++)
                {
                    TestingInput[i][j] /= 255;
                }
            }
        }

        public void SaveDataset(string train)
        {
            using (var writer = new System.IO.StreamWriter(train))
            {
                for (int i = 0; i < TestingLable.Count; i++)
                {
               
                    for (int j = 0; j < 10; j++)
                    {
                        if (TestingLable[i][0] == j)
                            writer.Write("1");
                        else
                            writer.Write("0");

                        writer.Write(",");

                    }
                    for (int k = 0; k < 28*28; k++)
                    {
                        writer.Write(TestingInput[i][k]);

                        if (k == 28 * 28 - 1)
                            continue;
                        writer.Write(",");
                    }
                    if (i == TestingLable.Count - 1)
                        continue;
                    writer.Write(Environment.NewLine);
                }
              
            }
            
        }
       

            

        
    }
}
