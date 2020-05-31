using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DNN
{
    class Neuron
    {
        #region Delegates
        private delegate double Activation_Functions(double Value);
        Activation_Functions Activation_Function;
        #endregion

        #region Parameters
        public enum ActivationFunction
        {
            Sigmoid,
            TanH,
            ReLU,
            LeakyReLU,
            BinaryStep,
            Softmax,

        };
        #endregion

        public double[] Neurons;

        public Neuron (int number, ActivationFunction activation_function)
        {
            Neurons = new double [number];

            switch (activation_function)
            {

                case ActivationFunction.Sigmoid:
                    Activation_Function = Sigmoid;                
                    break;
                case ActivationFunction.TanH:
                    Activation_Function = TanH;
                    break;
                case ActivationFunction.ReLU:
                    Activation_Function = ReLU;
                    break;
                case ActivationFunction.LeakyReLU:
                    Activation_Function = LeakyReLU;
                    break;
                case ActivationFunction.BinaryStep:
                    Activation_Function = BinaryStep;
                    break;
                default:
                    throw new ArgumentException("Not exict");

            }
        }
        public void Activate ()
        {
            for (int i = 0; i < Neurons.Length; i++)
            {
                Neurons[i] = Activation_Function(Neurons[i]);
            }
        }
        
        #region Activation Functions
        private double Sigmoid(double NeuralValue)
        {
            double ActivationValue = (1 / (1 + Math.Exp(-NeuralValue))) + Math.Exp(-745);
            return ActivationValue;
        }
        private double TanH(double NeuralValue)
        {
            double ActivationValue = Math.Tanh(NeuralValue) + Math.Exp(-745);
            return ActivationValue;
        }
        private double ReLU(double NeuralValue)
        {
            if (NeuralValue > 0)
                return NeuralValue;
            else
                return 0;
        }
        private double LeakyReLU(double NeuralValue)
        {
            if (NeuralValue > 0)
                return NeuralValue;
            else
                return (0.01 * NeuralValue);
        }
        private double BinaryStep(double NeuralValue)
        {
            if (NeuralValue > 0)
                return 1;
            else
                return 0;
        }

        private int SoftmaxIndexer = 0;
        private double SoftmaxSum = 0;
        private double Softmax(double NeuralValue)
        {
            double ActivationValue = Math.Exp(NeuralValue) + Math.Exp(-745);//softmax

            SoftmaxSum += ActivationValue;
            SoftmaxIndexer++;

            if (SoftmaxIndexer == Neurons.Length)
            {
                for (int i = 1; i < Neurons.Length; i++)//start from 1 to not recalculate the last neural twice
                {
                    Neurons[Neurons.Length - 1 - i] /= SoftmaxSum;//softmax
                }
                double SoftmaxSumBuffer = SoftmaxSum;
                SoftmaxIndexer = 0;
                SoftmaxSum = 0;

                return (ActivationValue / SoftmaxSumBuffer);
            }
            return ActivationValue;//softmax
        }
        #endregion

       

    }
}
