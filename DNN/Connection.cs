using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DNN
{

    class Connection
    {
        private Layer Input_Layer;
        private Layer Output_Layer;

        private double[] Weight;//weight array is one dimention for easy fast saving
        private double[] Bias;

        private int Weight_Length;
        private int Bias_Length;

        public Connection(Layer input_layer,Layer output_layer)
        {
            Input_Layer = input_layer;//get input layer
            Output_Layer = output_layer;//get output layer

            Weight_Length = Output_Layer.Neurons_Length * Input_Layer.Neurons_Length;//get weight array length
            Bias_Length = Output_Layer.Neurons_Length;//get bias array length

            Weight = new double[Weight_Length];//create weight array 
            Bias = new double[Bias_Length];//create bias array

            Random rand = new Random();//Initialize random weights and biases
            for (int i = 0; i < Weight_Length; i++)
                Weight[i] = 2;
            //Weight[i] = (double)rand.Next(-200, 200) / 1000;//get random from -0.2 to 0.2

            for (int i = 0; i < Bias_Length; i++)
                Bias[i] = 0;
               // Bias[i] = (double)rand.Next(-200, 200) / 1000;//end Initializing


        }
        public void FeedForward()
        {
            int WeightIndex = 0;//weight indexer

            for (int i = 0; i < Output_Layer.Neurons_Length; i++)
            {
                double Z = 0;
                for (int j = 0; j < Input_Layer.Neurons_Length; j++)
                {
                    Z += Input_Layer[j] * Weight[WeightIndex];//first index of output neurons = sum of input layer * weigts
                    WeightIndex++;
                }
                Output_Layer[i] = Z + Bias[i];//set output layer neuron and it will activate automatically

            }

        }
    }
}
