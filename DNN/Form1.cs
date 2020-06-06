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
        Dataset D;
     

        private void Form1_Load(object sender, EventArgs e)
        {

            Layer[] L = new Layer[4];
            L[0] = new Layer(784, Layer.ActivationFunction.ReLU);
            L[1] = new Layer(200, Layer.ActivationFunction.ReLU);
            L[2] = new Layer(80, Layer.ActivationFunction.ReLU);
            L[3] = new Layer(10, Layer.ActivationFunction.Sigmoid);

            Random rand = new Random();
            Connection[] C = new Connection[3];
            C[0] = new Connection(L[0], L[1], rand);
            C[1] = new Connection(L[1], L[2], rand);
            C[2] = new Connection(L[2], L[3], rand);

            M = new Model(L, C, Model.CostFunctions.MeanSquareSrror);
            D = new Dataset(784, 10);
            D.LoadDataset("mnist_test1.csv");




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

        }

        private void button1_Click(object sender, EventArgs e)
        {
            double[] Input_Layer = new double[2];
            Input_Layer[0] = double.Parse(textBox1.Text);
            Input_Layer[1] = double.Parse(textBox2.Text);

            double[] Output_Layer = M.FeedForward(Input_Layer);
            textBox3.Text = Output_Layer[0].ToString();
        }

        private void button2_Click(object sender, EventArgs e)
        {
            new Thread(() =>
            {
                Thread.CurrentThread.IsBackground = true;
                double Error = 1;
                while (Error > 0.001f)
                {
                    Error = LeanANDGate();
                }
                MessageBox.Show("done");

            }).Start();

        }
        private void button3_Click(object sender, EventArgs e)
        {
            new Thread(() =>
            {
                Thread.CurrentThread.IsBackground = true;
                double Error = 1;
                while (Error > 0.001f)
                {
                    Error = LeanORGate();
                }
                MessageBox.Show("done");

            }).Start();
        }
        private void button4_Click(object sender, EventArgs e)
        {
            new Thread(() =>
            {
                Thread.CurrentThread.IsBackground = true;
                double Error = 1;
                while (Error > 0.001f)
                {
                    Error = LeanXORGate();
                }
                MessageBox.Show("done");

            }).Start();
        }

        private double LeanANDGate()
        {
            double[] Input_Layer = new double[2];
            double[] Target_Output_Layer = new double[1];
            double Error = 0;
            Input_Layer[0] = 0;
            Input_Layer[1] = 0;
            Target_Output_Layer[0] = 0;

            Error += M.BackPropagation(Input_Layer, Target_Output_Layer);

            Input_Layer[0] = 1;
            Input_Layer[1] = 1;
            Target_Output_Layer[0] = 1;

            Error += M.BackPropagation(Input_Layer, Target_Output_Layer);

            Input_Layer[0] = 1;
            Input_Layer[1] = 0;
            Target_Output_Layer[0] = 0;

            Error += M.BackPropagation(Input_Layer, Target_Output_Layer);

            Input_Layer[0] = 0;
            Input_Layer[1] = 1;
            Target_Output_Layer[0] = 0;

            Error += M.BackPropagation(Input_Layer, Target_Output_Layer);

            Error /= 4;
            return Error;
            //double[] Output_Layer = M.FeedForward(Input_Layer);
            //textBox3.Text = Output_Layer[0].ToString();
        }
        private double LeanORGate()
        {
            double[] Input_Layer = new double[2];
            double[] Target_Output_Layer = new double[1];
            double Error = 0;
            Input_Layer[0] = 0;
            Input_Layer[1] = 0;
            Target_Output_Layer[0] = 0;

            Error += M.BackPropagation(Input_Layer, Target_Output_Layer);

            Input_Layer[0] = 1;
            Input_Layer[1] = 1;
            Target_Output_Layer[0] = 1;

            Error += M.BackPropagation(Input_Layer, Target_Output_Layer);

            Input_Layer[0] = 1;
            Input_Layer[1] = 0;
            Target_Output_Layer[0] = 1;

            Error += M.BackPropagation(Input_Layer, Target_Output_Layer);

            Input_Layer[0] = 0;
            Input_Layer[1] = 1;
            Target_Output_Layer[0] = 1;

            Error += M.BackPropagation(Input_Layer, Target_Output_Layer);

            Error /= 4;
            return Error;
            //double[] Output_Layer = M.FeedForward(Input_Layer);
            //textBox3.Text = Output_Layer[0].ToString();
        }
        private double LeanXORGate()
        {
            double[] Input_Layer = new double[2];
            double[] Target_Output_Layer = new double[1];
            double Error = 0;
            Input_Layer[0] = 0;
            Input_Layer[1] = 0;
            Target_Output_Layer[0] = 0;

            Error += M.BackPropagation(Input_Layer, Target_Output_Layer);

            Input_Layer[0] = 1;
            Input_Layer[1] = 1;
            Target_Output_Layer[0] = 0;

            Error += M.BackPropagation(Input_Layer, Target_Output_Layer);

            Input_Layer[0] = 1;
            Input_Layer[1] = 0;
            Target_Output_Layer[0] = 1;

            Error += M.BackPropagation(Input_Layer, Target_Output_Layer);

            Input_Layer[0] = 0;
            Input_Layer[1] = 1;
            Target_Output_Layer[0] = 1;

            Error += M.BackPropagation(Input_Layer, Target_Output_Layer);

            Error /= 4;
            return Error;
            //double[] Output_Layer = M.FeedForward(Input_Layer);
            //textBox3.Text = Output_Layer[0].ToString();
        }

        private void button5_Click(object sender, EventArgs e)
        {
            new Thread(() =>
            {
                for (int i = 0; i < 1000; i++)
                {
                    double Error = M.Train(D);
                    Invoke(new Action(() =>
                    {
                        label1.Text = Error.ToString();
                    }));
                }



            }).Start();


        }

        private void button6_Click(object sender, EventArgs e)
        {

        }
    }

}
