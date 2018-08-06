using Encog.Neural.Networks;
using Encog.Neural.Networks.Layers;
using Encog.Engine.Network.Activation;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Encog.ML.Train;
using Encog.Neural.Networks.Training.Propagation.Back;
using Encog.ML.Data;
using Encog.Neural.Data.Basic;

namespace ProjectToadNN
{
    class Program
    {
        static void Main(string[] args)
        {
            /// LIGHT DIODE ARRAY INFO
            double[][] dataIn = {
                new double[3]
                {
                    0, 0.5, 0.75
                }
            };
            
            /// ACTUAL VOLUME
            double[][] dataOut = {
                new double[1]
                {
                    0
                }
            };

            /// SETUP: MAKE PUBLIC
            IMLDataSet mLDataSet = new BasicNeuralDataSet(dataIn, dataOut);
            IMLDataSet mLInput;
            IMLTrain mLTrain;
            BasicNetwork basicNetwork = new BasicNetwork();

            /// NEURAL NETWORK 3 > 5 > 5 > 5 > 1
            basicNetwork.AddLayer(new BasicLayer(new ActivationSigmoid(), true, 3));
            basicNetwork.AddLayer(new BasicLayer(new ActivationSigmoid(), true, 5));
            basicNetwork.AddLayer(new BasicLayer(new ActivationSigmoid(), true, 5));
            basicNetwork.AddLayer(new BasicLayer(new ActivationSigmoid(), true, 5));
            basicNetwork.AddLayer(new BasicLayer(new ActivationSigmoid(), true, 1));
            basicNetwork.Structure.FinalizeStructure();
            basicNetwork.Reset();

            mLTrain = new Backpropagation(basicNetwork, mLDataSet);

            /// TRAINING
            float numEpochs = 100;
            for (float i = 0; i <= numEpochs; i++)
            {
                /// PROGRESSBAR
                Console.Clear();
                for (int pi = 0; pi <= (i / numEpochs) * 50; pi++)
                {

                    Console.Write("-");
                }
                Console.WriteLine("> {0}%", (float)((i/numEpochs) * 100), numEpochs);

                /// ITERATION
                mLTrain.Iteration();
            }

            /// INPUT
            while (true) { 
                var in1 = Console.ReadLine();
                var in2 = Console.ReadLine();
                var in3 = Console.ReadLine();

                double[][] inputIn =
                {
                    new double[3]
                    {
                        double.Parse(in1), double.Parse(in2), double.Parse(in3)
                    }
                };

                mLInput = new BasicNeuralDataSet(inputIn, dataOut);
                IMLData output = basicNetwork.Compute(mLInput[0].Input);

                Console.WriteLine((float)output[0]);
            }
        }
    }
}
