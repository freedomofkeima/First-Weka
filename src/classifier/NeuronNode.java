package classifier;

import helper.RandomHelper;

/**
 * Neuron Node for Neural Network Implementation
 * 
 * @author WbTeladan
 * @version 0.1, by WbTeladan @since October 15, 2014
 *
 */

public class NeuronNode {
	
	private double weight[];
	private double previous_weight[];
	private double output;
	private double error_ratio;
	private double bias;
	
	/**
	 * activation_type = 1 -> sigmoid (default)
	 * activation_type = 2 -> hardlim / step
	 * activation_type = 3 -> hardlims
	 * activation_type = 4 -> purelin
	 * 
	 */
	private int activation_type = 1;
	
	/** Input layer */
	public NeuronNode() {
		weight = null;
		previous_weight = null;
	}
	
	/** Randomize weight (Default) */
	public NeuronNode(int prev_size) {
		weight = new double[prev_size];
		previous_weight = new double[prev_size];
		for (int i = 0; i < prev_size; i++) {
			weight[i] = RandomHelper.randomValue();
			previous_weight[i] = weight[i];	
		}
		output = 0;
		bias = RandomHelper.randomValue();
		error_ratio = 0;
	}
	
	/** Custom weight */
	public NeuronNode(int prev_size, double w[]) {
		weight = new double[prev_size];
		previous_weight = new double[prev_size];
		for (int i = 0; i < prev_size; i++) {
			weight[i] = w[i];
			previous_weight[i] = weight[i];
		}
		output = 0;
		bias = RandomHelper.randomValue();
		error_ratio = 0;
	}

	public double getWeight(int idx) {
		return weight[idx];
	}

	public void setWeight(int idx, double weight) {
		this.weight[idx] = weight;
	}

	public double getPrevious_weight(int idx) {
		return previous_weight[idx];
	}

	public void setPrevious_weight(int idx, double previous_weight) {
		this.previous_weight[idx] = previous_weight;
	}
	
	public double getOutput() {
		return output;
	}

	public void setOutput(double output) {
		this.output = output;
	}
	
	public double getError_ratio() {
		return error_ratio;
	}

	public void setError_ratio(double error_ratio) {
		this.error_ratio = error_ratio;
	}

	public double getBias() {
		return bias;
	}

	public void setBias(double bias) {
		this.bias = bias;
	}

	public int getActivation_type() {
		return activation_type;
	}

	public void setActivation_type(int activation_type) {
		this.activation_type = activation_type;
	}
	
}
