package classifier;

import java.util.Enumeration;

import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.NoSupportForMissingValuesException;
import weka.core.Utils;

/**
 * Custom ID3 Implementation (extends Classifier)
 * 
 * @author WbTeladan
 * @version 0.1, by WbTeladan @since September 27, 2014
 *
 */

public class myID3 extends Classifier {

	/** Serializable */
	private static final long serialVersionUID = 5616209221366222508L;

	/** The node's successors. */
	private myID3[] successors;

	/** Attribute used for splitting. */
	private Attribute attribute;

	/** Class value if node is leaf. */
	private double classValue;

	/** Class distribution if node is leaf. */
	private double[] distribution;

	/** Class attribute of dataset. */
	private Attribute classAttribute;

	/** Class attribute of dataset. */
	private String[] anotherCA;

	/** Information Gain */
	private double[] infoGains;

	/**
	 * Returns default capabilities of the classifier
	 *
	 * @return the capabilities of this classifier
	 */
	public Capabilities getCapabilities() {
		Capabilities result = super.getCapabilities();
		result.disableAll();

		// Accept Nominal Attributes (only)
		result.enable(Capability.NOMINAL_ATTRIBUTES);

		// Accept Nominal Class & Missing Class Values
		result.enable(Capability.NOMINAL_CLASS);
		result.enable(Capability.MISSING_CLASS_VALUES);

		// Minimum number of instances
		result.setMinimumNumberInstances(0);

		return result;
	}

	/**
	 * Builds Custom Id3 decision tree classifier
	 *
	 * @param data
	 *            the training data
	 * @exception Exception
	 *                if classifier can't be built successfully
	 */
	public void buildClassifier(Instances data) throws Exception {

		// check data capabilities
		getCapabilities().testWithFail(data);

		// remove instances with missing class (pre-process)
		data = new Instances(data);
		data.deleteWithMissingClass();

		// create a new Tree from training data
		makeTree(data);
	}

	/**
	 * Method for building a custom Id3 tree
	 *
	 * @param data
	 *            the training data
	 * @exception Exception
	 *                if decision tree can't be built successfully
	 */
	@SuppressWarnings("rawtypes")
	private void makeTree(Instances data) throws Exception {

		// Check if no instances have reached this node.
		if (data.numInstances() == 0) {
			attribute = null;
			classValue = Instance.missingValue();
			distribution = new double[data.numClasses()];
			return;
		}

		// Compute attribute with maximum information gain.
		// double[] infoGains = new double[data.numAttributes()];

		infoGains = new double[data.numAttributes()];
		anotherCA = new String[data.numAttributes()];
		int count = 0;

		Enumeration attEnum = data.enumerateAttributes();
		while (attEnum.hasMoreElements()) {
			Attribute att = (Attribute) attEnum.nextElement();
			anotherCA[count] = att.name();
			infoGains[att.index()] = computeInfoGain(data, att);
			count++;
		}
		attribute = data.attribute(Utils.maxIndex(infoGains));// attribute with Max IG value

		// Make leaf if information gain is zero.
		// Otherwise create successors.
		if (Utils.eq(infoGains[attribute.index()], 0)) {
			attribute = null;
			distribution = new double[data.numClasses()];
			Enumeration instEnum = data.enumerateInstances();
			while (instEnum.hasMoreElements()) {
				Instance inst = (Instance) instEnum.nextElement();
				distribution[(int) inst.classValue()]++;
			}
			Utils.normalize(distribution); // convert it into 0.0 to 1.0 ratio
			classValue = Utils.maxIndex(distribution);
			classAttribute = data.classAttribute();
		} else {
			Instances[] splitData = splitData(data, attribute);
			successors = new myID3[attribute.numValues()];
			for (int j = 0; j < attribute.numValues(); j++) {
				successors[j] = new myID3();
				successors[j].makeTree(splitData[j]);
			}
		}
	}

	/**
	 * Classifies a given test instance using the decision tree.
	 *
	 * @param instance
	 *            the instance to be classified
	 * @return the classification
	 * @throws NoSupportForMissingValuesException
	 *             if instance has missing values
	 */
	public double classifyInstance(Instance instance)
			throws NoSupportForMissingValuesException {

		if (instance.hasMissingValue()) {
			throw new NoSupportForMissingValuesException(
					"Id3: no missing values, " + "please.");
		}
		if (attribute == null) {
			return classValue;
		} else {
			System.out.println(attribute.name() + " = "
					+ attribute.value((int) instance.value(attribute)));
			return successors[(int) instance.value(attribute)]
					.classifyInstance(instance);
		}
	}

	/**
	 * Computes class distribution for instance using decision tree.
	 *
	 * @param instance
	 *            the instance for which distribution is to be computed
	 * @return the class distribution for the given instance
	 * @throws NoSupportForMissingValuesException
	 *             if instance has missing values
	 */
	public double[] distributionForInstance(Instance instance)
			throws NoSupportForMissingValuesException {

		if (instance.hasMissingValue()) {
			throw new NoSupportForMissingValuesException(
					"Id3: no missing values, " + "please.");
		}
		if (attribute == null) {
			return distribution;
		} else {
			return successors[(int) instance.value(attribute)]
					.distributionForInstance(instance);
		}
	}

	/**
	 * Prints the decision tree using the private toString method from below.
	 *
	 * @return a textual description of the classifier
	 */
	public String toString() {

		if ((distribution == null) && (successors == null)) {
			return "Id3: No model built yet.";
		}
		return "Id3" + toString(0);
	}

	/**
	 * Computes information gain for an attribute.
	 *
	 * @param data
	 *            the data for which info gain is to be computed
	 * @param att
	 *            the attribute
	 * @return the information gain for the given attribute and data
	 * @throws Exception
	 *             if computation fails
	 */
	private double computeInfoGain(Instances data, Attribute att)
			throws Exception {

		double infoGain = computeEntropy(data);
		Instances[] splitData = splitData(data, att);
		for (int j = 0; j < att.numValues(); j++) {
			if (splitData[j].numInstances() > 0) {
				infoGain -= ((double) splitData[j].numInstances() / (double) data
						.numInstances()) * computeEntropy(splitData[j]);
			}
		}
		return infoGain;
	}

	/**
	 * Computes the entropy of a dataset.
	 * 
	 * @param data
	 *            the data for which entropy is to be computed
	 * @return the entropy of the data's class distribution
	 * @throws Exception
	 *             if computation fails
	 */
	@SuppressWarnings("rawtypes")
	private double computeEntropy(Instances data) throws Exception {

		double[] classCounts = new double[data.numClasses()];
		Enumeration instEnum = data.enumerateInstances();
		while (instEnum.hasMoreElements()) {
			Instance inst = (Instance) instEnum.nextElement();
			classCounts[(int) inst.classValue()]++;
		}
		double entropy = 0;
		for (int j = 0; j < data.numClasses(); j++) {
			if (classCounts[j] > 0) {
				entropy -= classCounts[j] * Utils.log2(classCounts[j]);
			}
		}
		entropy /= (double) data.numInstances();
		return entropy + Utils.log2(data.numInstances());
	}

	/**
	 * Splits a dataset according to the values of a nominal attribute.
	 *
	 * @param data
	 *            the data which is to be split
	 * @param att
	 *            the attribute to be used for splitting
	 * @return the sets of instances produced by the split
	 */
	@SuppressWarnings("rawtypes")
	private Instances[] splitData(Instances data, Attribute att) {

		Instances[] splitData = new Instances[att.numValues()];
		for (int j = 0; j < att.numValues(); j++) {
			splitData[j] = new Instances(data, data.numInstances());
		}
		Enumeration instEnum = data.enumerateInstances();
		while (instEnum.hasMoreElements()) {
			Instance inst = (Instance) instEnum.nextElement();
			splitData[(int) inst.value(att)].add(inst);
		}
		for (int i = 0; i < splitData.length; i++) {
			splitData[i].compactify();
		}
		return splitData;
	}

	/**
	 * Outputs a tree at a certain level.
	 *
	 * @param level
	 *            the level at which the tree is to be printed
	 * @return the tree as string at the given level
	 */
	private String toString(int level) {

		StringBuffer text = new StringBuffer();

		if (attribute == null) {
			if (Instance.isMissingValue(classValue)) {
				text.append(": null");
			} else {
				text.append(" Kelas : "
						+ classAttribute.value((int) classValue) + " [LEAF]");
			}
		} else {

			for (int j = 0; j < attribute.numValues(); j++) {
				text.append("\n");
				for (int i = 0; i <= level; i++) {
					text.append("| \n");
				}
				for (int k = 0; k + 1 < anotherCA.length; k++) { // null prevention
					text.append(" " + anotherCA[k]);
					text.append(" (IG = " + infoGains[k] + ")\n");
				}
				text.append(" " + attribute.name() + " = " + attribute.value(j)
						+ " \n");
				text.append(successors[j].toString(level + 1) + "\n");
			}
		}
		return text.toString();
	}

}
