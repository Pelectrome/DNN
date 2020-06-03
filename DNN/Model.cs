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

        private Layer[] Layers;
        private Connection[] Connections;
        private int OLIndex;

        public Model(Layer[] layers, Connection[] connections, CostFunctions cost_function)
        {
            Layers = layers;
            Connections = connections;
            OLIndex = layers.Length - 1;

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
        public double[] FeedForward(double[] input)
        {
            Layers[0].SetLayer = input;
            for (int i = 0; i < Connections.Length; i++)
            {
                Connections[i].FeedForward();
            }
            double[] Output_Layer = Layers[OLIndex].GetLayer;
            
            return Output_Layer;
        }

        public void BackPropagation(double[] input, double[] target)
        {
            double[] Output_Layer = FeedForward(input);

            for (int i = 0; i < Output_Layer.Length; i++)
            {
                Layers[OLIndex].Delta[i] = Delta_OutputLayer(Output_Layer[i],target[i]);//set delta for output layer
            }

            foreach (var item in Connections)
            {
                item.BackPropagateDelta();
            }

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
