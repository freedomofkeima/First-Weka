package classifier;

import java.util.ArrayList;
import java.util.Enumeration;
import java.util.List;

import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Custom Neural Network Implementation (extends Classifier)
 * 
 * @author WbTeladan
 * @version 0.1, by WbTeladan @since October 15, 2014
 *
 */

@SuppressWarnings("serial")
public class myNN extends Classifier {
	
	/** List of Attributes */
	
	private double learning_rate = 0.4; // default learning rate
	private double momentum = 0.9; // default momentum
	private int max_epoch = 500; // default maximum epoch
	private int epoch = 0; // current epoch
	private double min_error = 0.05; // default minimum error ratio
	private List<NeuronLayer> layers; // layers in myNN
	
	/**
	 * mode = 1 -> min_error ( deltaMSE )
	 * mode = 2 -> max_epoch
	 * 
	 */
	private int mode = 1;
	
	/**
	 * activation_type = 1 -> sigmoid
	 * activation_type = 2 -> step
	 * activation_type = 3 -> hardlin
	 * 
	 */
	private int activation_type = 1;
	
	/**
	 * topology = 1 -> Single Perceptron
	 * topology = 2 -> Single Perceptron with Batch Gradient Descent
	 * topology = 3 -> Single Perceptron with Delta Rule
	 * topology = 4 -> Multilayer Perceptron with Back Propagation
	 * 
	 */
	private final int topology;
	
	public myNN(int _topology) {
		layers = new ArrayList<NeuronLayer>();
		topology = _topology;
	}
	
	public void addLayer(NeuronLayer l) throws Exception {
		if (layers.size() == 2 && topology != 4)
			throw new IllegalArgumentException("Single Perceptron Only!");
		layers.add(l);
	}

	@SuppressWarnings("rawtypes")
	@Override
	public void buildClassifier(Instances instances) throws Exception {
		// TODO Auto-generated method stub
		if (layers.size() < 2) 
			throw new IllegalArgumentException("You need to define input and output layer!");
		
		double mse = 1; // mean error
		
		switch(topology) {
		case 1:
			break;
		case 2:
			break;
		case 3:
			break;
		case 4:
			do {
				Enumeration e = instances.enumerateInstances();
				while (e.hasMoreElements()) {
					Instance inst = (Instance) e.nextElement();
					
					// Step 0: Initialize input per training case
					// Step 1 : Forward Pass
					double classification = classifyInstance(inst);
					System.out.println(inst.value(0) + " " + inst.value(1) + " " + inst.value(2) + " - Result: " + classification);

					// Step 2: Back Propagation, Calculate Error Ratio
					for (int j = 0; j < layers.get(layers.size() - 1).getN_node(); j++) {
						// Propagate output layer
						double error_ratio = layers.get(layers.size() - 1).getNodes(j).getOutput();
						error_ratio *= (1 - error_ratio);
						error_ratio *= (inst.value(inst.numAttributes() - 1) - layers.get(layers.size() - 1).getNodes(j).getOutput());
						// Update error_ratio
						layers.get(layers.size() - 1).getNodes(j).setError_ratio(error_ratio);
					}
					
					for (int i = layers.size() - 2; i > 0; i--) {
						for (int j = 0; j < layers.get(i).getN_node(); j++) {
							// Propagate hidden layer
							double error_ratio = layers.get(i).getNodes(j).getOutput();
							error_ratio *= (1 - error_ratio);
							double sum_propagate = 0;
							for (int k = 0; k < layers.get(i+1).getN_node(); k++) {
								sum_propagate += (layers.get(i+1).getNodes(k).getWeight(j) * layers.get(i+1).getNodes(k).getError_ratio());
							}
							error_ratio *= sum_propagate;
							// Update error_ratio
							layers.get(i).getNodes(j).setError_ratio(error_ratio);
						}
					}
					
					// Step 3: Update weight & bias
					for (int i = layers.size() - 1; i > 0; i--) {
						for (int j = 0; j < layers.get(i).getN_node(); j++) {
							for (int k = 0; k < layers.get(i-1).getN_node(); k++) {
								// Delta weight
								double delta_weight = learning_rate * layers.get(i).getNodes(j).getError_ratio();
								delta_weight *= layers.get(i-1).getNodes(k).getOutput();
								
								// Momentum
								double delta_momentum = (layers.get(i).getNodes(j).getWeight(k) 
										- layers.get(i).getNodes(j).getPrevious_weight(k));
								delta_momentum *= momentum;
								
								// Update weight
								double w = layers.get(i).getNodes(j).getWeight(k);
								layers.get(i).getNodes(j).setPrevious_weight(k, w);
								layers.get(i).getNodes(j).setWeight(k, w + delta_weight + delta_momentum);
								
								/*
								System.out.println(i + " " + j + " " + k + " " 
								+ layers.get(i).getNodes(j).getPrevious_weight(k) + " " 
								+ delta_weight + " "
								+ (w + delta_weight + delta_momentum));
								*/
								
								// Update bias
								double b = layers.get(i).getNodes(j).getBias();
								double delta_bias = learning_rate * layers.get(i).getNodes(j).getError_ratio();
								layers.get(i).getNodes(j).setBias(b + delta_bias);
							}
						}
					}
					
				}
				
				// re-calculate MSE
				mse = 0;
				e = instances.enumerateInstances();
				while (e.hasMoreElements()) {
					Instance inst = (Instance) e.nextElement();
					classifyInstance(inst);
					double val = inst.value(inst.numAttributes() - 1) - layers.get(layers.size() - 1).getNodes(0).getOutput();
					val *= val; // quadratic
					mse += val;
				}
				
				System.out.println(toString()); // output weight per epoch
				epoch++; // next epoch
			} while (epoch < max_epoch && mse > min_error);
			break;
		}
	}
	
	public double classifyInstance(Instance instance) throws Exception {
		
		double input[] = new double[layers.get(0).getN_node()];
		for (int i = 0; i < instance.numAttributes() - 1; i++) {
			input[i] = instance.value(i);
		}
		layers.get(0).setInputValue(input);
		
		for (int i = 1; i < layers.size(); i++) {
			for (int j = 0; j < layers.get(i).getN_node(); j++) {
				double net = forwardPass(layers.get(i).getNodes(j), layers.get(i-1));
				double output = activate(net);
				layers.get(i).getNodes(j).setOutput(output); // update output per node
			}
		}
		
		double result = layers.get(layers.size() - 1).getNodes(0).getOutput();
		if (result + 1e-9 > 0.5) return 1;
		else return 0;
	}
	
	public String toString() {
		System.out.println("--Epoch " + epoch + "--");
		
		// Print weights here
		for (int i = 1; i < layers.size(); i++) {
			System.out.print("Layer " + i + ": ");
			for (int j = 0; j < layers.get(i).getN_node(); j++) {
				for (int k = 0; k < layers.get(i-1).getN_node(); k++) {
					System.out.print(layers.get(i).getNodes(j).getWeight(k) + " ");
				}
			}
			System.out.println();
		}
		
		// Print additional information here
		
		return "";
	}
	
	private double forwardPass(NeuronNode node, NeuronLayer prev_layer) {
		double net = 0;
		
		for (int i = 0; i < prev_layer.getN_node(); i++) {
			net += node.getWeight(i) * prev_layer.getNodes(i).getOutput();
		}
		net += node.getBias();
		
		return net;
	}
	
	/** Activate */
	private double activate(double v) {
		switch (activation_type) {
		case 1: // sigmoid
			if (v > 45) return 1;
			if (v < -45) return 0;
			return (1 / (1 + Math.exp(-v)));
		case 2: // step
			
		case 3: // hardlin
			
		}
		return 0;
	}

	/** Getter & Setter for several parameters */
	public double getLearning_rate() {
		return learning_rate;
	}

	public void setLearning_rate(double learning_rate) {
		this.learning_rate = learning_rate;
	}

	public double getMomentum() {
		return momentum;
	}

	public void setMomentum(double momentum) {
		this.momentum = momentum;
	}

	public int getMax_epoch() {
		return max_epoch;
	}

	public void setMax_epoch(int max_epoch) {
		this.max_epoch = max_epoch;
	}

	public int getEpoch() {
		return epoch;
	}

	public void setEpoch(int epoch) {
		this.epoch = epoch;
	}

	public double getMin_error() {
		return min_error;
	}

	public void setMin_error(double min_error) {
		this.min_error = min_error;
	}

	public int getMode() {
		return mode;
	}

	public void setMode(int mode) {
		this.mode = mode;
	}

	public int getActivation_type() {
		return activation_type;
	}

	public void setActivation_type(int activation_type) {
		this.activation_type = activation_type;
	}
	
	public int getLayerSize() {
		if (layers.size() == 0) return 0;
		else return layers.get(layers.size() - 1).getN_node();
	}

}
