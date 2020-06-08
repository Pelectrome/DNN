using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace DNN
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
        }
       
        Model M;
        NeuralNetwork NNTEST;
        Dataset D;
     

        private void Form1_Load(object sender, EventArgs e)
        {

            Layer[] L1 = new Layer[2];
            L1[0] = new Layer(784, Layer.ActivationFunction.ReLU);
            L1[1] = new Layer(200, Layer.ActivationFunction.ReLU);

            Layer[] L2 = new Layer[2];
            L2[0] = new Layer(80, Layer.ActivationFunction.ReLU);
            L2[1] = new Layer(10, Layer.ActivationFunction.Softmax);

            LConnection[] C1 = new LConnection[1];
            C1[0] = new LConnection(L1[0], L1[1]);

            LConnection[] C2 = new LConnection[1];
            C2[0] = new LConnection(L2[0], L2[1]);



            NeuralNetwork[] NN = new NeuralNetwork[2];
            NN[0] = new NeuralNetwork(L1, C1,0.01);
            NN[1] = new NeuralNetwork(L2, C2,0.01);

            NNConnection[] NNC = new NNConnection[1];
            NNC[0] = new NNConnection(NN[0], NN[1]);

            M = new Model(NN, NNC, Model.CostFunctions.CrossEntropy,0.01);

            D = new Dataset(784, 10);
            D.LoadDataset("mnist_test.csv");




            //string w = null;

            //    for (int i = 0; i < C[0].WeightBackMap.Length; i++)
            //    {
            //        w += C[0].WeightBackMap[i] + ",";
            //    }
            //for (int i = 0; i < C[1].WeightBackMap.Length; i++)
            //{
            //    w += (C[1].WeightBackMap[i]+25) + ",";
            //}

            //w += "\r\n";
            //w += "\r\n";
            //for (int i = 0; i < NN._WeightsBackMap.Length; i++)
            //{
            //    w+=NN._WeightsBackMap[NN._WeightsBackMap.Length-i-1] +",";
            //}
            //MessageBox.Show(w);
            Layer[] LTEST = new Layer[4];
            LTEST[0] = new Layer(784, Layer.ActivationFunction.ReLU);
            LTEST[1] = new Layer(200, Layer.ActivationFunction.ReLU);
            LTEST[2] = new Layer(80, Layer.ActivationFunction.ReLU);
            LTEST[3] = new Layer(10, Layer.ActivationFunction.Softmax);
            LConnection[] CTEST = new LConnection[3];
            CTEST[0] = new LConnection(LTEST[0], LTEST[1]);
            CTEST[1] = new LConnection(LTEST[1], LTEST[2]);
            CTEST[2] = new LConnection(LTEST[2], LTEST[3]);

            NNTEST = new NeuralNetwork(LTEST, CTEST, NeuralNetwork.CostFunctions.CrossEntropy, 0.01);

        }

    
        private void button5_Click(object sender, EventArgs e)
        {
            new Thread(() =>
            {
                for (int i = 0; i < 1000; i++)
                {
                    double Error = NNTEST.Train(D);
                    Invoke(new Action(() =>
                    {
                        label1.Text = Error.ToString();
                    }));
                }



            }).Start();
            new Thread(() =>
            {
                for (int i = 0; i < 1000; i++)
                {
                    double Error = M.Train(D);
                    Invoke(new Action(() =>
                    {
                        label2.Text = Error.ToString();
                    }));
                }



            }).Start();


        }

        private void button6_Click(object sender, EventArgs e)
        {

        }
    }

}
