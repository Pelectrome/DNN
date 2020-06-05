using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
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
            L[0]=new Layer(2, Layer.ActivationFunction.Sigmoid);
            L[1] = new Layer(2, Layer.ActivationFunction.Sigmoid);
            L[2]=new Layer(1, Layer.ActivationFunction.Sigmoid);
        
            Connection[] C = new Connection[2];
            C[0] = new Connection(L[0], L[1]);
            C[1] = new Connection(L[1], L[2]);

            M = new Model(L, C,Model.CostFunctions.MeanSquareSrror);

           // MessageBox.Show(L[0].Delta[0].ToString());
        }

        private void button1_Click(object sender, EventArgs e)
        {
            if (!isDouble(textBox1.Text) || !isDouble(textBox2.Text))
            {
                MessageBox.Show("invalid input");
                return;
            }
            double[] Input_Layer = new double[2];
            Input_Layer[0] = double.Parse(textBox1.Text);
            Input_Layer[1] = double.Parse(textBox2.Text);

            double[] Output_Layer = M.FeedForward(Input_Layer);
            textBox3.Text = Output_Layer[0].ToString();
        }
        
        bool OnTime = true;
        private void button2_Click(object sender, EventArgs e)
        {
            
            timer1.Enabled = OnTime;
            OnTime = !OnTime;

        }

        private void timer1_Tick(object sender, EventArgs e)
        {
            LeanANDGate();
        }
        private void LeanANDGate()
        {
            double[] Input_Layer = new double[2];
            double[] Target_Output_Layer = new double[1];

            Input_Layer[0] = 0;
            Input_Layer[1] = 0;
            Target_Output_Layer[0] = 1;

            M.BackPropagation(Input_Layer, Target_Output_Layer);

            Input_Layer[0] = 1;
            Input_Layer[1] = 1;
            Target_Output_Layer[0] = 1;

            M.BackPropagation(Input_Layer, Target_Output_Layer);

            Input_Layer[0] = 1;
            Input_Layer[1] = 0;
            Target_Output_Layer[0] = 0;

            M.BackPropagation(Input_Layer, Target_Output_Layer);

            Input_Layer[0] = 0;
            Input_Layer[1] = 1;
            Target_Output_Layer[0] = 0;

            M.BackPropagation(Input_Layer, Target_Output_Layer);


            double[] Output_Layer = M.FeedForward(Input_Layer);
            textBox3.Text = Output_Layer[0].ToString();
        }
        bool isDouble(string str)
        {
            return double.TryParse(str, out _);
        }
    }
    
}
