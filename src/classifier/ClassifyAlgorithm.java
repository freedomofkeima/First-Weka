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
	 * @param trainingSet
	 * @return
	 * @throws Exception
	 */
	public static Classifier naiveBayesAlgorithm(Instances trainingSet, int id) throws Exception {
		 // Create a naive bayes classifier 
		 Classifier cModel = (Classifier) new NaiveBayes();
		 if (id == 1) {
			 Evaluation eval = new Evaluation(trainingSet);
			 eval.crossValidateModel(cModel, trainingSet, 10, new Random(1));
			 System.out.println("Percent correct: "+
                     Double.toString(eval.pctCorrect()));
		 }
		 cModel.buildClassifier(trainingSet);
		 return cModel;
	}
	
	public static Classifier iD3Algorithm(Instances trainingSet, int id) throws Exception {
		// Create a ID3 classifier
		Classifier cModel = (Classifier) new Id3();
		if (id == 1) {
			 Evaluation eval = new Evaluation(trainingSet);
			 eval.crossValidateModel(cModel, trainingSet, 10, new Random(1));
			 System.out.println("Percent correct: "+
                    Double.toString(eval.pctCorrect()));
		}
		cModel.buildClassifier(trainingSet);
		return cModel;
	}

}
