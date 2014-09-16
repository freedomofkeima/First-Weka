package core;

import filter.SupervisedFilter;
import helper.Constants;
import helper.TextWriter;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.Enumeration;

import classifier.ClassifyAlgorithm;
import loader.LoadARFF;
import loader.LoadCSV;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Attribute;
import weka.core.Instances;

/**
 * Main class for Weka
 * 
 * @author Iskandar Setiadi
 * @version 0.1, by IS @since September 16, 2014
 *
 */

public class Main {

	@SuppressWarnings("unchecked")
	public static void main(String[] args) throws Exception {

		Instances data = null;
		Classifier cModel = null;
		String input, input2;

		BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
		while (true) {
			TextWriter.printMainMenu();
			input = reader.readLine();

			switch (input) {
			case "1":
				/** Load from .arff */
				TextWriter.printLoadMenu();
				input2 = reader.readLine();
				if (input2.equals("1")) {
					data = LoadARFF.loadARFF(Constants.ARFF_NOMINAL_PATH);
				} else if (input2.equals("2")) {
					data = LoadARFF.loadARFF(Constants.ARFF_NUMERIC_PATH);
				}
				// Set play {yes, no}
				data.setClassIndex(data.numAttributes() - 1);
				break;
			case "2":
				/** Load from .csv */
				TextWriter.printLoadMenu();
				input2 = reader.readLine();
				if (input2.equals("1")) {
					data = LoadCSV.loadCSV(Constants.CSV_NOMINAL_PATH);
				} else if (input2.equals("2")) {
					data = LoadCSV.loadCSV(Constants.CSV_NUMERIC_PATH);
				}
				// Set play {yes, no}
				data.setClassIndex(data.numAttributes() - 1);
				break;
			case "3":
				/** Remove attribute (outlook) */
				if (data != null) {
					Enumeration<Attribute> e;

					e = data.enumerateAttributes();
					TextWriter.printEnumerationAttribute(e);
					// Delete first attribute - Outlook
					data.deleteAttributeAt(0);

					e = data.enumerateAttributes();
					TextWriter.printEnumerationAttribute(e);
				} else {
					System.out.println("You need to load your data first!");
				}
				break;
			case "4":
				/** Filter (resample) */
				if (data != null) {
					System.out.println("# Previous : " + data.numInstances());
					data = SupervisedFilter.resampleInstances(data);
					System.out.println("# After : " + data.numInstances());
				} else {
					System.out.println("You need to load your data first!");
				}
				break;
			case "5":
				/** Build classifier with Naive Bayes */
				if (data != null) {
					TextWriter.printClassifierMenu();
					input2 = reader.readLine();
					if (input2.equals("1")) {
						cModel = ClassifyAlgorithm.naiveBayesAlgorithm(data, 1);
					} else if (input2.equals("2")) {
						cModel = ClassifyAlgorithm.naiveBayesAlgorithm(data, 2);
					}
				} else {
					System.out.println("You need to load your data first!");
				}
				break;
			case "6":
				/** Build classifier with DT */
				if (data != null) {
					TextWriter.printClassifierMenu();
					input2 = reader.readLine();
					if (input2.equals("1")) {
						cModel = ClassifyAlgorithm.iD3Algorithm(data, 1);
					} else if (input2.equals("2")) {
						cModel = ClassifyAlgorithm.iD3Algorithm(data, 2);
					}
				} else {
					System.out.println("You need to load your data first!");
				}
				break;
			case "7":
				/** Testing model given test set (Assume train = test) */
				if (cModel != null) {
					Evaluation eval = new Evaluation(data);
					eval.evaluateModel(cModel, data);
					System.out.println(eval.toSummaryString(
							"\nResults\n======\n", false));
				} else {
					System.out.println("You need to build classifier first!");
				}
				break;
			case "8":
				/** Testing model to classify one unseen data */
				if (cModel != null) {

				} else {
					System.out.println("You need to build classifier first!");
				}
				break;
			case "9":
				/** Save model */
				if (cModel != null) {

				} else {
					System.out.println("You need to build classifier first!");
				}
				break;
			case "10":
				/** Load model */

				break;
			case "11":
				/** Classify using extended classifier */
				if (cModel != null) {

				} else {
					System.out.println("You need to build classifier first!");
				}
				break;
			case "999":
				System.out.println("Goodbye!");
				return;
			default:
				System.out.println("Unrecognized input value!");
			}
		}

	}

}
