using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DNN
{
    class Model
    {
        private delegate double Cost_Functions_Derivative(double neural, double output);
        Cost_Functions_Derivative Delta_OutputLayer;

        public enum CostFunctions
        {
            MeanSquareSrror,
            MeanAbsoluteError,
            CrossEntropy
        };

        private NeuralNetwork[] NeuralNetworks;
        private NNConnection[] NNConnections;

        private int ONNIndex;//Output Neural Network Index
        private int OLIndex;//Output layer Index of output model

        public Model(NeuralNetwork[] neural_networks, NNConnection[] neural_networks_Connections, CostFunctions cost_function, double learing_rate = 0.1)
        {
            NeuralNetworks = neural_networks;
            NNConnections = neural_networks_Connections;

            ONNIndex = NeuralNetworks.Length - 1;
            OLIndex = NeuralNetworks[ONNIndex].OLIndex;

            switch (cost_function)
            {
                case CostFunctions.MeanSquareSrror:

                    if (NeuralNetworks[ONNIndex].Layers[OLIndex].GetActivatonFunction() == Layer.ActivationFunction.Sigmoid)
                        Delta_OutputLayer = DeltaMeanSquareErrorSigmoid;

                    else if (NeuralNetworks[ONNIndex].Layers[OLIndex].GetActivatonFunction() == Layer.ActivationFunction.TanH)
                        Delta_OutputLayer = DeltaMeanSquareErrorTanH;

                    break;
                case CostFunctions.MeanAbsoluteError:

                    if (NeuralNetworks[ONNIndex].Layers[OLIndex].GetActivatonFunction() == Layer.ActivationFunction.Sigmoid)
                        Delta_OutputLayer = DeltaMeanAbsoluteErrorSigmoid;

                    else if (NeuralNetworks[ONNIndex].Layers[OLIndex].GetActivatonFunction() == Layer.ActivationFunction.TanH)
                        Delta_OutputLayer = DeltaMeanAbsoluteErrorTanH;

                    break;
                case CostFunctions.CrossEntropy:

                    if (NeuralNetworks[ONNIndex].Layers[OLIndex].GetActivatonFunction() == Layer.ActivationFunction.Sigmoid)
                        Delta_OutputLayer = DeltaCorssEntorpySigmoid;

                    else if (NeuralNetworks[ONNIndex].Layers[OLIndex].GetActivatonFunction() == Layer.ActivationFunction.TanH)
                        Delta_OutputLayer = DeltaCorssEntorpyTanH;

                    else if (NeuralNetworks[ONNIndex].Layers[OLIndex].GetActivatonFunction() == Layer.ActivationFunction.Softmax)
                        Delta_OutputLayer = DeltaCorssEntorpySoftmax;

                    break;
                default:
                    throw new ArgumentException("Cost function can't be set");
            }

            foreach (var item in NNConnections)
            {
                item.Model_Connection_LearningRate = learing_rate;
            }
        }
        public double[] FeedForward(double[] input)
        {
            NeuralNetworks[0].Layers[0].SetLayer = input;

            for (int i = 0; i < NNConnections.Length; i++)
            {
                NNConnections[i].FeedForward();
            }
            double[] Output_Layer = NeuralNetworks[ONNIndex].Layers[OLIndex].GetLayer;

            return Output_Layer;
        }
        public double BackPropagation(double[] input, double[] target)
        {
            double[] Output_Layer = FeedForward(input);
            double Error = 0;

            for (int i = 0; i < Output_Layer.Length; i++)
            {
                NeuralNetworks[ONNIndex].Layers[OLIndex].Delta[i] = Delta_OutputLayer(Output_Layer[i], target[i]);//set delta for output layer
                Error += Math.Abs(Output_Layer[i] - target[i]);
            }
            for (int i = 0; i < NNConnections.Length; i++)
            {
                NNConnections[NNConnections.Length - 1 - i].BackPropagateDelta();
            }
            foreach (var item in NNConnections)
            {
                item.UpdateWeights();
            }
            Error /= Output_Layer.Length;
            return Error;

        }

        public double Train(Dataset dataset)
        {
            int Dataset_Length = dataset.Length / 100;
            double Error = 0;
            for (int i = 0; i < Dataset_Length; i++)
            {
                Error += BackPropagation(dataset.InputDataset[i], dataset.LableDataset[i]);
            }
            Error /= Dataset_Length;
            return Error;
        }
        #region Cost Functions
        private double DeltaMeanSquareErrorSigmoid(double Neural, double Target)
        {
            return ((Neural - Target) * Neural * (1 - Neural));
        }
        private double DeltaMeanAbsoluteErrorSigmoid(double Neural, double Target)
        {
            return (Math.Sign(Neural - Target) * Neural * (1 - Neural));
        }
        private double DeltaCorssEntorpySigmoid(double Neural, double Target)
        {
            return (Neural - Target);
        }
        private double DeltaMeanSquareErrorTanH(double Neural, double Target)
        {
            return ((Neural - Target) * (1 - Neural * Neural));
        }
        private double DeltaMeanAbsoluteErrorTanH(double Neural, double Target)
        {
            return (Math.Sign(Neural - Target) * (1 - Neural * Neural));
        }
        private double DeltaCorssEntorpyTanH(double Neural, double Target)
        {
            return (Neural - Target);
        }
        private double DeltaCorssEntorpySoftmax(double Neural, double Target)
        {
            return (Neural - Target);
        }
        #endregion
    }
}
