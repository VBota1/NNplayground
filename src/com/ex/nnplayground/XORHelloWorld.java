package com.ex.nnplayground;

import android.app.Activity;
import android.os.Bundle;
import android.text.method.ScrollingMovementMethod;
import android.widget.TextView;

import org.encog.Encog;
import org.encog.engine.network.activation.ActivationBiPolar;
import org.encog.engine.network.activation.ActivationClippedLinear;
import org.encog.engine.network.activation.ActivationElliott;
import org.encog.engine.network.activation.ActivationFunction;
import org.encog.engine.network.activation.ActivationGaussian;
import org.encog.engine.network.activation.ActivationLinear;
import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.engine.network.activation.ActivationSteepenedSigmoid;
import org.encog.engine.network.activation.ActivationStep;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;
 
/**
 * XOR: This example is essentially the "Hello World" of neural network
 * programming.  This example shows how to construct an Encog neural
 * network to predict the output from the XOR operator.  This example
 * uses backpropagation to train the neural network.
 * 
 * This example attempts to use a minimum of Encog features to create and
 * train the neural network.  This allows you to see exactly what is going
 * on.  For a more advanced example, that uses Encog factories, refer to
 * the XORFactory example.
 * 
 */
public class XORHelloWorld extends Activity {

	/**
	 * The input necessary for XOR.
	 */
	public static double XOR_INPUT[][] = { { 0.0, 0.0 }, { 1.0, 0.0 },
			{ 0.0, 1.0 }, { 1.0, 1.0 } };
	
	public static double XOR_SECOND_INPUT[][] = { { 0.0 }, { 0.0 }, { 0.0 }, { 0.0 } };
 
	/**
	 * The ideal data necessary for XOR.
	 */
	public static double XOR_IDEAL[][] = { { 1.0 }, { 0.0 }, { 0.0 }, { 1.0 } };

	@Override
	public void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);
		
		
	    // Get the message from the intent
	    //Intent intent = getIntent();
	    //String message = intent.getStringExtra(NNMainActivity.EXTRA_MESSAGE);
	    
		//create view to display results
	    TextView textView = new TextView(this);
	    textView.setMovementMethod(new ScrollingMovementMethod());
	    //textView.setTextSize(10);		
	    textView.setText("NEWRAL NETWORK THAT SOLVES XOR");
		
		// create a neural network, without using a factory
		BasicNetwork network = new BasicNetwork();
		network.addLayer(new BasicLayer(null,false,2));
		network.addLayer(new BasicLayer(new ActivationSigmoid(),true,3));
		network.addLayer(new BasicLayer(new ActivationSigmoid(),false,1));
		//network.addLayer(new BasicLayer(new ActivationSigmoid(),false,1));		
		network.getStructure().finalizeStructure();
		network.reset();
		/*
		//network.setLayerBiasActivation(3, 0.5);
		network.setWeight(0, 0, 0, -1.36);
		network.setWeight(0, 1, 0, -0.836);
		network.setWeight(0, 0, 1, -9.16);
		network.setWeight(0, 1, 1, 4.53);
		network.setWeight(0, 0, 2, 2.89);
		network.setWeight(0, 1, 2, -12.14);		
		network.setWeight(1, 0, 0, -9.17);
		network.setWeight(1, 1, 0, 3.7);
		network.setWeight(1, 2, 0, 3.37);
		//network.setWeight(2, 0, 0, 2);		
*/
		// create training data
		MLDataSet trainingSet = new BasicMLDataSet(XOR_INPUT, XOR_IDEAL);
 
		// train the neural network
		final ResilientPropagation train = new ResilientPropagation(network, trainingSet);
 
		int epoch = 1;

	    //train
	    textView.append("\n"); //new line	    
	    textView.append("Trainig information:________________________________________");
		do {
			train.iteration();
			epoch++;
		} while(train.getError() > 0.0001);
		train.finishTraining();
		
		//write details about last training operation
	    textView.append("\n"); //new line		
		textView.append("Epoch #" + epoch + " Error:" + train.getError());

   
		//blank line
	    textView.append("\n"); //new line		
	    
	    // test the neural network
		textView.append("Test network:________________________________________");
		int i=0;
		for(MLDataPair pair: trainingSet ) {
			final MLData output = network.compute(pair.getInput());
		    textView.append("\n"); //new line	
			textView.append("expected: " + pair.getInput().getData(0) + " XOR " + pair.getInput().getData(1) + "=" + pair.getIdeal().getData(0));
		    textView.append("\n"); //new line			
			textView.append("actual: " + pair.getInput().getData(0) + " XOR " + pair.getInput().getData(1) + "=" + output.getData(0));
			
			XOR_SECOND_INPUT[i][0] = output.getData(0);
			i++;
		}
		
	    textView.append("\n"); //new line	
	    //textView.append("i="+XOR_SECOND_INPUT[1][0]);
	    textView.append("XOR_SECOND_INPUT ="+"\n"+"["+XOR_SECOND_INPUT[0][0]+"\n"+XOR_SECOND_INPUT[1][0]+"\n"+XOR_SECOND_INPUT[2][0]+"\n"+XOR_SECOND_INPUT[3][0]+"]");
	    
		// create a neural network, without using a factory
		BasicNetwork network2 = new BasicNetwork();
		network2.addLayer(new BasicLayer(null,false,1));
		//network2.addLayer(new BasicLayer(new ActivationLinear(),false,2));
		network2.addLayer(new BasicLayer(new ActivationClippedLinear(),true,1));
		//network.addLayer(new BasicLayer(new ActivationSigmoid(),false,1));		
		network2.getStructure().finalizeStructure();
		network2.reset();	    
		
		// create training data
		MLDataSet trainingSet2 = new BasicMLDataSet(XOR_SECOND_INPUT, XOR_IDEAL);		

		// train the neural network
		final ResilientPropagation train2 = new ResilientPropagation(network2, trainingSet2);
 
		int epoch2 = 1;

	    //train
	    textView.append("\n"); //new line	    
	    textView.append("Trainig information:________________________________________");
		do {
			train2.iteration();
			epoch2++;
		} while(train2.getError() > 0.005);
		train2.finishTraining();		
		
		//write details about last training operation
	    textView.append("\n"); //new line		
		textView.append("Epoch #" + epoch2 + " Error:" + train2.getError());

   
		//blank line
	    textView.append("\n"); //new line		

	    // test the neural network
		textView.append("Test network:________________________________________");
		for(MLDataPair pair: trainingSet2 ) {
			final MLData output = network2.compute(pair.getInput());
		    textView.append("\n"); //new line	
			textView.append("expected: " + pair.getInput().getData(0) + "=" + pair.getIdeal().getData(0));
		    textView.append("\n"); //new line			
			textView.append("actual: " + pair.getInput().getData(0) + "=" + output.getData(0));
		}

/*
		textView.append("\n"); //new line		
		textView.append("Display network:________________________________________");
		int layerCount=network.getLayerCount();
		int[] neuronCount = { 0, 0, 0 };
		int i,j,k;
	    textView.append("\n"); //new line		
		textView.append("number of layers="+layerCount);
		for (i=0;i<layerCount; i++)
		{
			neuronCount[i]=network.getLayerNeuronCount(i);
		    textView.append("\n"); //new line		
			textView.append("number of neurons in layer "+i+" ="+neuronCount[i]);
		}
		
		for (i=0;i<(layerCount-1); i++)
		{
			for (j=0; j<neuronCount[i]; j++)
				for (k=0; k<neuronCount[i+1]; k++)
				{
				    textView.append("\n"); //new line		
					textView.append("layer "+i+" neuron "+j+" to"+" layer "+(i+1)+" neuron "+k+" weight= "+network.getWeight(i, j, k));
				}
		}
	*/
		
		Encog.getInstance().shutdown();
	   
	    // Set the text view as the activity layout
	    setContentView(textView);	

	}

}
