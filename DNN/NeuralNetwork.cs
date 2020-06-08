using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DNN
{
    class NeuralNetwork
    {
        private delegate double Cost_Functions_Derivative(double neural, double output);
        Cost_Functions_Derivative Delta_OutputLayer;

        public enum CostFunctions
        {
            MeanSquareSrror,
            MeanAbsoluteError,
            CrossEntropy
        };

        public Layer[] Layers { get; }
        public int OLIndex { get; }

        private LConnection[] LConnections;
       

        public NeuralNetwork(Layer[] layers, LConnection[] layers_connections, CostFunctions cost_function,double learing_rate)
        {
            Layers = layers;
            LConnections = layers_connections;
            OLIndex = layers.Length - 1;

            foreach (var item in LConnections)
            {
                item.LearningRate = learing_rate;
            }

            switch (cost_function)
            {
                case CostFunctions.MeanSquareSrror:

                    if (Layers[OLIndex].GetActivatonFunction() == Layer.ActivationFunction.Sigmoid)
                        Delta_OutputLayer = DeltaMeanSquareErrorSigmoid;

                    else if (Layers[OLIndex].GetActivatonFunction() == Layer.ActivationFunction.TanH)
                        Delta_OutputLayer = DeltaMeanSquareErrorTanH;

                    break;
                case CostFunctions.MeanAbsoluteError:

                    if (Layers[OLIndex].GetActivatonFunction() == Layer.ActivationFunction.Sigmoid)
                        Delta_OutputLayer = DeltaMeanAbsoluteErrorSigmoid;

                    else if (Layers[OLIndex].GetActivatonFunction() == Layer.ActivationFunction.TanH)
                        Delta_OutputLayer = DeltaMeanAbsoluteErrorTanH;

                    break;
                case CostFunctions.CrossEntropy:

                    if (Layers[OLIndex].GetActivatonFunction() == Layer.ActivationFunction.Sigmoid)
                        Delta_OutputLayer = DeltaCorssEntorpySigmoid;

                    else if (Layers[OLIndex].GetActivatonFunction() == Layer.ActivationFunction.TanH)
                        Delta_OutputLayer = DeltaCorssEntorpyTanH;

                    else if (Layers[OLIndex].GetActivatonFunction() == Layer.ActivationFunction.Softmax)
                        Delta_OutputLayer = DeltaCorssEntorpySoftmax;

                    break;
                default:
                    throw new ArgumentException("Cost function can't be set");
            }
        }
        public NeuralNetwork(Layer[] layers, LConnection[] layers_connections, double learing_rate = 0.1)
        {
            Layers = layers;
            LConnections = layers_connections;
            OLIndex = layers.Length - 1;

            foreach (var item in LConnections)
            {
                item.LearningRate = learing_rate;
            }
        }

        public double[] FeedForward(double[] input)
        {
            Layers[0].SetLayer = input;

            for (int i = 0; i < LConnections.Length; i++)
            {
                LConnections[i].FeedForward();
            }
            double[] Output_Layer = Layers[OLIndex].GetLayer;

            return Output_Layer;
        }
        public void FeedForward()
        {
            for (int i = 0; i < LConnections.Length; i++)
            {
                LConnections[i].FeedForward();
            }

        }

        public void BackPropagateDelta()
        {
            for (int i = 0; i < LConnections.Length; i++)//from output layer to input layer
            {
                LConnections[LConnections.Length - 1 - i].BackPropagateDelta();
            }
        }
        public void UpdateWeights()
        {
            foreach (var item in LConnections)
            {
                item.UpdateWeights();
            }

        }

        public double BackPropagation(double[] input, double[] target)
        {
            double[] Output_Layer = FeedForward(input);
            double Error = 0;

            for (int i = 0; i < Output_Layer.Length; i++)
            {
                Layers[OLIndex].Delta[i] = Delta_OutputLayer(Output_Layer[i], target[i]);//set delta for output layer
                Error += Math.Abs(Output_Layer[i] - target[i]);
            }
            for (int i = 0; i < LConnections.Length; i++)
            {
                LConnections[LConnections.Length - 1 - i].BackPropagateDelta();
            }

            foreach (var item in LConnections)
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
