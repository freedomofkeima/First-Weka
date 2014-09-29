package classifier;

import java.util.ArrayList;
import java.util.Enumeration;
import java.util.List;

import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

/**
 * Custom ID3 Implementation (extends Classifier)
 * 
 * @author WbTeladan
 * @version 0.1, by WbTeladan @since September 28, 2014
 *
 */

@SuppressWarnings("serial")
public class myID3 extends Classifier {

	private static double EMPTY_VALUE = Instance.missingValue();
	private static Attribute node_attribute;
	private static String[] attribute_title;

	private double classification;
	private Attribute node_value;
	private boolean isChecked = false;
	private myID3[] next_node;
	private double[] IG;

	@SuppressWarnings("rawtypes")
	@Override
	public void buildClassifier(Instances arg0) throws Exception {
		Capabilities check_cap = super.getCapabilities();
		check_cap.disableAll();
		check_cap.enable(Capability.NOMINAL_ATTRIBUTES);
		check_cap.enable(Capability.NOMINAL_CLASS);
		check_cap.enable(Capability.MISSING_CLASS_VALUES);
		check_cap.setMinimumNumberInstances(0);
		check_cap.testWithFail(arg0);

		arg0.deleteWithMissingClass();

		if (attribute_title == null) {
			int count = 0;
			attribute_title = new String[arg0.numAttributes()];
			Enumeration e = arg0.enumerateAttributes();
			while (e.hasMoreElements()) {
				attribute_title[count] = ((Attribute) e.nextElement()).name();
				count++;
			}
		}
		node_attribute = arg0.classAttribute();

		process(arg0);
	}

	@SuppressWarnings("rawtypes")
	private void process(Instances arg0) throws Exception {
		if (arg0.numInstances() == 0) {
			classification = EMPTY_VALUE;
			node_value = null;
			isChecked = true;
			return;
		} else {
			IG = new double[arg0.numAttributes()];

			Enumeration e = arg0.enumerateAttributes();
			while (e.hasMoreElements()) {
				double information_gain = entropyCalculation(arg0);
				Attribute att = (Attribute) e.nextElement();
				Instances[] s = ntr(arg0, att);
				int total_instances = arg0.numInstances();

				for (int i = 0; i < att.numValues(); i++) {
					int n = s[i].numInstances();
					if (n > 0) {
						information_gain -= ((double) n / (double) total_instances)
								* entropyCalculation(s[i]);
					}
				}
				/** Information Gain */
				IG[att.index()] = information_gain;
			}

			int max_index = 0;
			for (int i = 1; i < arg0.numAttributes(); i++) {
				if (IG[i] > IG[max_index]) {
					max_index = i;
				}
			}
			node_value = arg0.attribute(max_index);

			if (IG[node_value.index()] == 0) {
				/** Leaf */
				double[] num_of_elements = new double[arg0.numClasses()];
				isChecked = true;
				node_value = null;

				Enumeration e2 = arg0.enumerateInstances();
				while (e2.hasMoreElements()) {
					Instance inst = (Instance) e2.nextElement();
					int class_value = (int) inst.classValue();
					num_of_elements[class_value]++;
				}
				Utils.normalize(num_of_elements);

				int max_index2 = 0;
				for (int i = 1; i < arg0.numClasses(); i++) {
					if (num_of_elements[i] > num_of_elements[max_index]) {
						max_index = i;
					}
				}

				classification = num_of_elements[max_index2];
			} else {
				/** Recursive */
				next_node = new myID3[node_value.numValues()];
				Instances[] s = ntr(arg0, node_value);
				for (int i = 0; i < node_value.numValues(); i++) {
					next_node[i] = new myID3();
					next_node[i].process(s[i]);
				}
			}

		}
	}

	@SuppressWarnings("rawtypes")
	private double entropyCalculation(Instances arg0) throws Exception {
		double return_value = 0;

		for (int i = 0; i < arg0.numClasses(); i++) {
			int num_of_elements = 0;

			Enumeration e = arg0.enumerateInstances();
			while (e.hasMoreElements()) {
				Instance inst = (Instance) e.nextElement();
				if (inst.classValue() == i) {
					num_of_elements++;
				}
			}

			if (num_of_elements > 0) {
				double v = (double) num_of_elements / arg0.numInstances();
				return_value -= Utils.log2(v) * v;
			}
		}
		return return_value;
	}

	@SuppressWarnings("rawtypes")
	private Instances[] ntr(Instances arg0, Attribute a) {
		Instances[] return_value = new Instances[a.numValues()];

		for (int i = 0; i < a.numValues(); i++) {
			List<Instance> l = new ArrayList<Instance>();

			Enumeration e = arg0.enumerateInstances();
			while (e.hasMoreElements()) {
				Instance inst = (Instance) e.nextElement();
				if ((int) inst.value(a) == i) {
					l.add(inst);
				}
			}

			return_value[i] = new Instances(arg0, l.size());
			for (Instance inst : l) {
				return_value[i].add(inst);
			}
		}

		return return_value;
	}

	public double classifyInstance(Instance arg0) throws Exception {
		double return_value = EMPTY_VALUE;
		if (node_value == null)
			return_value = classification;
		else {
			System.out.println(node_value.name() + " = "
					+ node_value.value((int) arg0.value(node_value)));
			return_value = next_node[(int) arg0.value(node_value)]
					.classifyInstance(arg0);
		}
		return return_value;
	}

	public String toString() {
		String results = "-- ID3 Model --";
		if (!isChecked && (next_node == null))
			return "Tree is Empty";
		return results + recursive_print(0);
	}

	public String recursive_print(int level) {
		String results = "";

		if (node_value == null) {
			if (Instance.isMissingValue(classification)) {
				results = results + " null";
			} else {
				results = results + " Kelas : "
						+ node_attribute.value((int) classification)
						+ " [LEAF]";
			}
		} else {

			for (int j = 0; j < node_value.numValues(); j++) {
				results = results + "\n";
				for (int i = 0; i <= level; i++) {
					results = results + "| \n";
				}
				for (int k = 0; k + 1 < attribute_title.length; k++) { // null
																		// prevention
					results = results + " " + attribute_title[k] + "";
					results = results + " (IG = " + IG[k] + ")\n";
				}
				results = results + " " + node_value.name() + " = "
						+ node_value.value(j) + " \n";
				results = results + next_node[j].recursive_print(level + 1)
						+ "\n";
			}
		}

		return results;
	}

}
