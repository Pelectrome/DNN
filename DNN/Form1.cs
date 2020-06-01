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

        private void Form1_Load(object sender, EventArgs e)
        {
           

            Layer[] L = new Layer[3];
            L[0]=new Layer(2, Layer.ActivationFunction.Sigmoid);
            L[1]=new Layer(2, Layer.ActivationFunction.Sigmoid);
            L[2]=new Layer(1, Layer.ActivationFunction.Sigmoid);

            Connection[] C = new Connection[2];
            C[0] = new Connection(L[0], L[1]);
            C[1] = new Connection(L[1], L[2]);
            for (int i = 0; i < C.Length; i++)
            {
                C[i].FeedForward();
            }

            MessageBox.Show(L[2][0].ToString());
        }

  
    }
}
