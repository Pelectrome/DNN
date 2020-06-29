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
        NeuralNetwork[] NN = new NeuralNetwork [2];
        Dataset D0;
        Dataset D1;
        private List<double[]> Input = new List<double[]>();
        private void Form1_Load(object sender, EventArgs e)
        {

            Input.Add(new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 });
            Input.Add(new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 1, 0 });
            Input.Add(new double[] { 0, 0, 0, 0, 0, 0, 0, 1, 0, 0 });
            Input.Add(new double[] { 0, 0, 0, 0, 0, 0, 1, 0, 0, 0 });
            Input.Add(new double[] { 0, 0, 0, 0, 0, 1, 0, 0, 0, 0 });
            Input.Add(new double[] { 0, 0, 0, 0, 1, 0, 0, 0, 0, 0 });
            Input.Add(new double[] { 0, 0, 0, 1, 0, 0, 0, 0, 0, 0 });
            Input.Add(new double[] { 0, 0, 1, 0, 0, 0, 0, 0, 0, 0 });
            Input.Add(new double[] { 0, 1, 0, 0, 0, 0, 0, 0, 0, 0 });
            Input.Add(new double[] { 1, 0, 0, 0, 0, 0, 0, 0, 0, 0 });

            //Layer[] L1 = new Layer[2];
            //L1[0] = new Layer(784, Layer.ActivationFunction.ReLU);
            //L1[1] = new Layer(200, Layer.ActivationFunction.ReLU);

            //Layer[] L2 = new Layer[2];
            //L2[0] = new Layer(80, Layer.ActivationFunction.ReLU);
            //L2[1] = new Layer(10, Layer.ActivationFunction.Softmax);

            //LConnection[] C1 = new LConnection[1];
            //C1[0] = new LConnection(L1[0], L1[1]);

            //LConnection[] C2 = new LConnection[1];
            //C2[0] = new LConnection(L2[0], L2[1]);



            //NeuralNetwork[] NN = new NeuralNetwork[2];
            //NN[0] = new NeuralNetwork(L1, C1,0.01);
            //NN[1] = new NeuralNetwork(L2, C2,0.01);

            //NNConnection[] NNC = new NNConnection[1];
            //NNC[0] = new NNConnection(NN[0], NN[1]);

            //M = new Model(NN, NNC, Model.CostFunctions.CrossEntropy,0.01);

            D0 = new Dataset(784, 10);
            D0.LoadTesting("mnist_test.csv");
            D0.LoadTraining("mnist_test.csv");


            D1 = new Dataset(10, 10);
            D1.TrainingInput = Input;
            D1.TrainingLable = Input;

            Layer[] L0 = new Layer[2];
            L0[0] = new Layer(10, Layer.ActivationFunction.Sigmoid);
            L0[1] = new Layer(20, Layer.ActivationFunction.Sigmoid);

            Random rand = new Random();
            LConnection[] C0 = new LConnection[1];
            C0[0] = new LConnection(L0[0], L0[1], rand);


            NN[0] = new NeuralNetwork(L0, C0, 0.1);


            Layer[] L1 = new Layer[3];
            L1[0] = new Layer(784, Layer.ActivationFunction.ReLU);
            L1[1] = new Layer(200, Layer.ActivationFunction.ReLU);
            L1[2] = new Layer(10, Layer.ActivationFunction.Softmax);
            LConnection[] C1 = new LConnection[2];
            C1[0] = new LConnection(L1[0], L1[1], rand);
            C1[1] = new LConnection(L1[1], L1[2], rand);

            NN[1] = new NeuralNetwork(L1, C1, NeuralNetwork.CostFunctions.CrossEntropy, 0.01);


           

            NNConnection[] NNC = new NNConnection[1];
            NNC[0]=new NNConnection( NN[0], NN[1] );
            M = new Model(NN, NNC, Model.CostFunctions.MeanSquareSrror, 0.1);
        }

       
        private void button5_Click(object sender, EventArgs e)
        {
            timer1.Enabled = true;
            new Thread(() =>
            {
               // NN[1].LearningRate = 0.001;

                for (int i = 0; i < 1; i++)
                {
                    double Error = NN[1].Train(D0);
                    timer1.Enabled = false;

                    double time = (double)TimeCounter / 10;
                    MessageBox.Show(string.Format("Time spend = {0}s",time));

                    double[] Input_Layer = NN[1].FeedBackward(Input[index]);

                    //string s = null;
                    //double[] Ourput_Layer1 = NN[1].FeedForward(D0.InputDataset[index]);
                    //for (int j = 0; j < Ourput_Layer1.Length; j++)
                    //{
                    //    s += Ourput_Layer1[j].ToString() + Environment.NewLine;
                    //}
                 

                    int ILInderxer = 0;
                    for (int j = 0; j < 28; j++)
                    {
                        for (int k = 0; k < 28; k++)
                        {
                            byte R = (byte)(Input_Layer[ILInderxer] * 255);
                            byte G = (byte)(Input_Layer[ILInderxer] * 255);
                            byte B = (byte)(Input_Layer[ILInderxer] * 255);

                            bmp.SetPixel(k, j, Color.FromArgb(R, G, B));
                            ILInderxer++;
                        }
                    }
                    Buffer = Zoom(bmp, 10);
                    //Buffer.Save(i.ToString() + ".jpg");
                    Invoke(new Action(() =>
                   {
                    label2.Text = Error.ToString();
                    pictureBox3.Image = Buffer;
                       //label4.Text = s;
                   }));
        }
                MessageBox.Show("done");
            }).Start();


        }

        Bitmap bmp = new Bitmap(28, 28);
        Bitmap Buffer = new Bitmap(280, 280);
        int index =  0;
        private void GetImage()
        {
           

            string s = null;

            //double[] Ourput_Layer1 = NN[1].FeedForward(D0.InputDataset[index]);
            //for (int i = 0; i < Ourput_Layer1.Length; i++)
            //{
            //    s += Ourput_Layer1[i].ToString() + Environment.NewLine;
            //}
            //label4.Text = s;


            double[] Ourput_Layer2 = M.FeedForward(D1.TrainingLable[index]);
            s = null;
            for (int i = 0; i < Ourput_Layer2.Length; i++)
            {
                s += Ourput_Layer2[i].ToString() + Environment.NewLine;
            }
            label3.Text = s;
            s = null;
            for (int i = 0; i < Ourput_Layer2.Length; i++)
            {
                s += D1.TrainingLable[index][i].ToString() + Environment.NewLine;
            }
            label5.Text = s;


            double[] Ourput_Layer = NN[1].Layers[0].GetLayer;
            int OLInderxer = 0;
            for (int i = 0; i < 28; i++)
            {
                for (int j = 0; j < 28; j++)
                {
                    byte R = (byte)(Ourput_Layer[OLInderxer] * 255);
                    byte G = (byte)(Ourput_Layer[OLInderxer] * 255);
                    byte B = (byte)(Ourput_Layer[OLInderxer] * 255);

                    bmp.SetPixel(j, i, Color.FromArgb(R, G, B));
                    OLInderxer++;
                }
            }
            Buffer = Zoom(bmp, 10);

                pictureBox3.Image = Buffer;
      
        }
        private void button2_Click(object sender, EventArgs e)
        {
            index++;
            if (index == 10)
                index = 0;
            GetImage();
            Buffer.Save(index.ToString() + ".jpg");
        }

        static public Bitmap Zoom(Bitmap image, float Scaling)
        {
            int Width = (int)Math.Round(image.Width * Scaling);
            int Height = (int)Math.Round(image.Height * Scaling);

            Bitmap bmp = new Bitmap(Width, Height);

            Graphics graphics = Graphics.FromImage(bmp);
            graphics.Clear(Color.Black);

            graphics.CompositingQuality = System.Drawing.Drawing2D.CompositingQuality.HighQuality;
            graphics.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.NearestNeighbor;
            graphics.PixelOffsetMode = System.Drawing.Drawing2D.PixelOffsetMode.HighSpeed;
            graphics.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.None;
            graphics.DrawImage(image, new Rectangle(0, 0, bmp.Width+10, bmp.Height+10));

            graphics.Dispose();

            return bmp;
        }
        private void button1_Click(object sender, EventArgs e)
        {
    

            new Thread(() =>
            {
                NN[1].LearningRate = 0;

                for (int i = 0; i < 10000; i++)
                {
                    double Error = M.Train(D1);

                    Invoke(new Action(() =>
                    {
                        GetImage();
                        label1.Text = Error.ToString();
                    }));
                }

            }).Start();
        }

        private void button3_Click(object sender, EventArgs e)
        {
           double Error = NN[1].ClassificationTest(D0,0.8)*100;
            MessageBox.Show(Error.ToString());
           
        }
        int TimeCounter = 0;
        private void timer1_Tick(object sender, EventArgs e)
        {
            TimeCounter++;
        }
    }

}
