package classifier;

import java.util.Enumeration;

import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Custom Algorithm Implementation (extends Classifier)
 * This file is modified from 
 * http://cns-classes.bu.edu/cn710/uploads/Main/WekaNotes.pdf
 * 
 * @author Iskandar Setiadi
 * @version 0.1, by IS @since September 16, 2014
 *
 */

@SuppressWarnings("serial")
public class CustomAlgorithm extends Classifier {

	Instances m_Instances;
	int m_nAttributes;

	@Override
	public void buildClassifier(Instances data) throws Exception {
		// TODO Auto-generated method stub
		m_Instances = new Instances(data);
		m_nAttributes = data.numAttributes();
	}

	@Override
	public double classifyInstance(Instance instance) {
		@SuppressWarnings("rawtypes")
		Enumeration enu = m_Instances.enumerateInstances();
		double distance = 9999999;
		double classValue = -1;
		while (enu.hasMoreElements()) {
			Instance _instance = (Instance) enu.nextElement();
			double _distance = CalculateDistance(instance, _instance);
			if (_distance < distance) {
				distance = _distance;
				classValue = _instance.classValue();
			}
		}
		return classValue;
	}

	public double CalculateDistance(Instance i1, Instance i2) {
		double s = 0;
		for (int i = 0; i < m_nAttributes - 1; i++) {
			double p = (i1.value(i) - i2.value(i));
			s += p * p;
		}
		return s;
	}

	@Override
	public String[] getOptions() {
		String[] options = new String[2];
		int current = 0;
		while (current < options.length)
			options[current++] = "";
		return options;
	}

}
