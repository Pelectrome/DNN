using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DNN
{
    class Layer
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

        #region Global Variables
        private double[] Neurons;
        private double[] Neurons_Buffer;

        public double this[int i]
        {
            get { return Neurons[i]; }
            set { Neurons[i] = Activation_Function(value); }//activate neuron automatically
        }
        public double[] SetLayer
        {
            set { value.CopyTo(Neurons,0); }//set layer value
        }
        public double[] GetLayer
        {
            get { Neurons.CopyTo(Neurons_Buffer, 0); return Neurons_Buffer; }//get layer value
        }

        public double[] Delta { get; set; }
        public int Neurons_Length { get; }
        
        #endregion

        #region Constructors
        public Layer(int number, ActivationFunction activation_function)
        {  
            Neurons_Length = number;
            Neurons = new double [Neurons_Length];
            Neurons_Buffer = new double[Neurons_Length];
            Delta = new double[Neurons_Length];

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
                case ActivationFunction.Softmax:
                    Activation_Function = Softmax;
                    break;
                default:
                    throw new ArgumentException("Not Exist");

            }
        }
        #endregion

        #region Methods
        public ActivationFunction GetActivatonFunction()
        {
            if (Activation_Function == Sigmoid)
                return ActivationFunction.Sigmoid;

            else if (Activation_Function == TanH)
                return ActivationFunction.TanH;

            else if (Activation_Function == ReLU)
                return ActivationFunction.ReLU;

            else if (Activation_Function == LeakyReLU)
                return ActivationFunction.LeakyReLU;

            else if (Activation_Function == BinaryStep)
                return ActivationFunction.BinaryStep;

            else if (Activation_Function == Softmax)
                return ActivationFunction.Softmax;

            throw new ArgumentException("Not Exist");        

        }
        #endregion

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
                for (int i = 0; i < Neurons.Length - 1; i++)
                {
                    Neurons[i] /= SoftmaxSum;//softmax
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
