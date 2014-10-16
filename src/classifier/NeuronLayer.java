package classifier;


/**
 * Neuron Layer for Neural Network Implementation
 * 
 * @author WbTeladan
 * @version 0.1, by WbTeladan @since October 15, 2014
 *
 */

public class NeuronLayer {
	private int n_node;
	private NeuronNode nodes[];
	
	/** Random Weight */
	public NeuronLayer(int prev_size, int node_size) {
		n_node = node_size;
		nodes = new NeuronNode[node_size];
		
		for (int i = 0; i < n_node; i++) {
			if (prev_size != 0) nodes[i] = new NeuronNode(prev_size);
			else nodes[i] = new NeuronNode();
		}
	}
	
	/** Custom Weight */
	public NeuronLayer(int prev_size, int node_size, double c_weights[][]) throws Exception {
		
		// Initial check to c_weights
		for (double[] weight_1 : c_weights) {
			for (double weight : weight_1) {
				if (weight > 1 || weight < -1)
					throw new IllegalArgumentException("Weight must between -1 to 1");	
			}
		}
		
		n_node = node_size;
		nodes = new NeuronNode[node_size];
		
		for (int i = 0; i < n_node; i++) {
			if (prev_size != 0)	nodes[i] = new NeuronNode(prev_size, c_weights[i]);
			else nodes[i] = new NeuronNode();
		}
	}

	public int getN_node() {
		return n_node;
	}

	public void setN_node(int n_node) {
		this.n_node = n_node;
	}

	public NeuronNode getNodes(int idx) {
		return nodes[idx];
	}

	public void setNodes(int idx, NeuronNode node) {
		this.nodes[idx] = node;
	}
	
	public void setInputValue(double o[]) {
		for (int i = 0; i < n_node; i++) {
			nodes[i].setOutput(o[i]);
		}
	}
	
}
