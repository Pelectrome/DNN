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
        Cost_Functions_Derivative Delta_InputLayer;

        public enum CostFunctions
        {
            MeanSquareSrror,
            MeanAbsoluteError,
            CrossEntropy
        };

        public Layer[] Layers { get; }
        public int OLIndex { get; }
        public double LearningRate
        {
            set
            {
                foreach (var item in LConnections)
                {
                    item.LearningRate = value;
                }
            }
        }

        private LConnection[] LConnections;


        public NeuralNetwork(Layer[] layers, LConnection[] layers_connections, CostFunctions cost_function, double learing_rate)
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
        public NeuralNetwork(Layer[] layers, LConnection[] layers_connections, CostFunctions output_cost_function, CostFunctions input_cost_function, double learing_rate)
        {
            Layers = layers;
            LConnections = layers_connections;
            OLIndex = layers.Length - 1;

            foreach (var item in LConnections)
            {
                item.LearningRate = learing_rate;
            }

            switch (input_cost_function)
            {
                case CostFunctions.MeanSquareSrror:

                    if (Layers[0].GetActivatonFunction() == Layer.ActivationFunction.Sigmoid)
                        Delta_InputLayer = DeltaMeanSquareErrorSigmoid;

                    else if (Layers[0].GetActivatonFunction() == Layer.ActivationFunction.TanH)
                        Delta_InputLayer = DeltaMeanSquareErrorTanH;

                    break;
                case CostFunctions.MeanAbsoluteError:

                    if (Layers[0].GetActivatonFunction() == Layer.ActivationFunction.Sigmoid)
                        Delta_InputLayer = DeltaMeanAbsoluteErrorSigmoid;

                    else if (Layers[0].GetActivatonFunction() == Layer.ActivationFunction.TanH)
                        Delta_InputLayer = DeltaMeanAbsoluteErrorTanH;

                    break;
                case CostFunctions.CrossEntropy:

                    if (Layers[0].GetActivatonFunction() == Layer.ActivationFunction.Sigmoid)
                        Delta_InputLayer = DeltaCorssEntorpySigmoid;

                    else if (Layers[0].GetActivatonFunction() == Layer.ActivationFunction.TanH)
                        Delta_InputLayer = DeltaCorssEntorpyTanH;

                    else if (Layers[0
                        ].GetActivatonFunction() == Layer.ActivationFunction.Softmax)
                        Delta_InputLayer = DeltaCorssEntorpySoftmax;

                    break;
                default:
                    throw new ArgumentException("Cost function can't be set");
            }
            switch (output_cost_function)
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
        public void UpdateWeights()
        {
            foreach (var item in LConnections)
            {
                item.UpdateWeights();
            }

        }

        public double[] FeedBackward(double[] output)
        {
            Layers[OLIndex].SetLayer = output;

            for (int i = 0; i < LConnections.Length; i++)
            {
                LConnections[LConnections.Length - 1 - i].FeedBackward();
            }
            double[] Input_Layer = Layers[0].GetLayer;

            return Input_Layer;
        }
        public void ForwardPropagateDelta()
        {
            for (int i = 0; i < LConnections.Length; i++)//from output layer to input layer
            {
                LConnections[i].BackPropagateDelta();
            }
        }
        public double ForwardPropagation(double[] output, double[] target)
        {
            double[] Input_Layer = FeedBackward(output);
            double Error = 0;

            for (int i = 0; i < Input_Layer.Length; i++)
            {
                Layers[0].Delta[i] = Delta_InputLayer(Input_Layer[i], target[i]);//set delta for output layer
                Error += Math.Abs(Input_Layer[i] - target[i]);
            }
            for (int i = 0; i < LConnections.Length; i++)
            {
                LConnections[i].ForwardPropagateDelta();
            }

            foreach (var item in LConnections)
            {
                item.UpdateWeightsBackward();
            }

            Error /= Input_Layer.Length;
            return Error;

        }
    
        public double Train(Dataset dataset)
        {
            int Training_Length = dataset.TrainingLength;
            double Error = 0;
            for (int i = 0; i < Training_Length; i++)
            {
                //Error += ForwardPropagation(dataset.TrainingLable[i], dataset.TrainingInput[i]);
                Error += BackPropagation(dataset.TrainingInput[i], dataset.TrainingLable[i]);            
            }
            Error /= Training_Length;
            return Error;
        }
        public double ClassificationTest(Dataset dataset, double threshold)
        {
            double[] Output;
            double Result = 0;
            int Testing_Length = dataset.TestingLength;

            for (int i = 0; i < Testing_Length; i++)
            {
                Output = FeedForward(dataset.TestingInput[i]);//get the output of the neural network

                double OutputMax = 0;
                int OutputMaxIndex = 0;
                int TestingLableMaxIndex = 0;

                for (int j = 0; j < Output.Length; j++)
                {
                    if (Output[j] > OutputMax)
                    {
                        OutputMax = Output[j];
                        OutputMaxIndex = j;
                    }
                    if ((int)dataset.TestingLable[i][j] == 1)
                        TestingLableMaxIndex = j;//get the index of the lable class
                }

                if (TestingLableMaxIndex == OutputMaxIndex)
                {
                    if (OutputMax > threshold)
                        Result++;
                }


            }
            Result /= Testing_Length;//get the average result for all testing data

            return Result;
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
