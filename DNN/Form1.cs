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
        private void Form1_Load(object sender, EventArgs e)
        {


            Layer[] L = new Layer[3];
            L[0] = new Layer(2, Layer.ActivationFunction.Sigmoid);
            L[1] = new Layer(2, Layer.ActivationFunction.Sigmoid);
            L[2] = new Layer(1, Layer.ActivationFunction.Sigmoid);

            Connection[] C = new Connection[2];
            C[0] = new Connection(L[0], L[1]);
            C[1] = new Connection(L[1], L[2]);

            M = new Model(L, C, Model.CostFunctions.MeanSquareSrror);
            Dataset D = new Dataset(10, 780);
            D.LoadDataset("mnist_test.csv");
            // MessageBox.Show(L[0].Delta[0].ToString());
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
                while (Error > 0.01f)
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
                while (Error > 0.01f)
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
                while (Error > 0.01f)
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


    }

}
