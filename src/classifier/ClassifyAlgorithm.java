package classifier;

import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.trees.Id3;
import weka.core.Instances;

/**
 * Algorithm Collection for Application
 * 
 * @author Iskandar Setiadi
 * @version 0.1, by IS @since September 16, 2014
 *
 */

public class ClassifyAlgorithm {

	/**
	 * Classifier Model using Naive Bayes Algorithm
	 * 
	 * @param trainingSet
	 * @param id
	 * @return
	 * @throws Exception
	 */
	public static Classifier naiveBayesAlgorithm(Instances trainingSet, int id)
			throws Exception {
		// Create a naive bayes classifier
		Classifier cModel = (Classifier) new NaiveBayes();
		if (id == 1) {
			Evaluation eval = new Evaluation(trainingSet);
			eval.crossValidateModel(cModel, trainingSet, 10, new Random(1));
			System.out.println(eval.toSummaryString("\nResults\n======\n", false));
			cModel.buildClassifier(trainingSet);
		} else if (id == 2) {
			/** Split is not random (Preserve Order for debug) */
			int trainSize = (int) Math.round(trainingSet.numInstances() * 0.5);
			int testSize = trainingSet.numInstances() - trainSize;
			Instances train = new Instances(trainingSet, 0, trainSize);
			Instances test = new Instances(trainingSet, trainSize, testSize);
			cModel.buildClassifier(train);

			/** Test section */
			Evaluation eval = new Evaluation(train);
			eval.evaluateModel(cModel, test);
			System.out.println(eval.toSummaryString("\nResults\n======\n", false));
		}
		return cModel;
	}

	/**
	 * Classifier Model using ID Tree Algorithm
	 * @param trainingSet
	 * @param id
	 * @return
	 * @throws Exception
	 */
	public static Classifier iD3Algorithm(Instances trainingSet, int id)
			throws Exception {
		// Create an ID3 classifier
		Classifier cModel = (Classifier) new Id3();
		if (id == 1) {
			Evaluation eval = new Evaluation(trainingSet);
			eval.crossValidateModel(cModel, trainingSet, 10, new Random(1));
			System.out.println(eval.toSummaryString("\nResults\n======\n", false));
			cModel.buildClassifier(trainingSet);
		} else if (id == 2) {
			/** Split is not random (Preserve Order for debug) */
			int trainSize = (int) Math.round(trainingSet.numInstances() * 0.5);
			int testSize = trainingSet.numInstances() - trainSize;
			Instances train = new Instances(trainingSet, 0, trainSize);
			Instances test = new Instances(trainingSet, trainSize, testSize);
			cModel.buildClassifier(train);

			/** Test section */
			Evaluation eval = new Evaluation(train);
			eval.evaluateModel(cModel, test);
			System.out.println(eval.toSummaryString("\nResults\n======\n", false));
		}
		return cModel;
	}

}
