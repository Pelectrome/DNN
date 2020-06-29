using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DNN
{
    class NNConnection
    {
        private NeuralNetwork Input_Neural_Nerwork;
        private NeuralNetwork Output_Neural_Nerwork;
        private LConnection Model_Connection;
        public double Model_Connection_LearningRate
        {
            set
            {
                Model_Connection.LearningRate = value;
            }
        }

        public NNConnection(NeuralNetwork input_neural_network,NeuralNetwork output_neural_network)
        {
            Input_Neural_Nerwork= input_neural_network;
            Output_Neural_Nerwork = output_neural_network;
            Random rand = new Random();

            Model_Connection = new LConnection(Input_Neural_Nerwork.Layers[Input_Neural_Nerwork.Layers.Length - 1], Output_Neural_Nerwork.Layers[0],rand);//connect output layer of first model to the input layer of second model

        }
        public void FeedForward()
        {
            Input_Neural_Nerwork.FeedForward();
            Model_Connection.FeedForward();
            Output_Neural_Nerwork.FeedForward();
        }
        public void BackPropagateDelta()
        {
            Output_Neural_Nerwork.BackPropagateDelta();
            Model_Connection.BackPropagateDelta();
            Input_Neural_Nerwork.BackPropagateDelta();
        }
        public void UpdateWeights()
        {
            Input_Neural_Nerwork.UpdateWeights();
            Model_Connection.UpdateWeights();
            Output_Neural_Nerwork.UpdateWeights();
        }
    }
}
