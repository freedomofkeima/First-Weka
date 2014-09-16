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

		BufferedReader reader = new BufferedReader(new InputStreamReader(
				System.in));
		while (true) {
			TextWriter.printMainMenu();
			input = reader.readLine();

			switch (input) {
			case "1":
				/** Load from .arff*/
				TextWriter.printLoadMenu();
				input2 = reader.readLine();
				if (input2.equals("1")) {
					data = LoadARFF.loadARFF(Constants.ARFF_NOMINAL_PATH);
				} else if (input2.equals("2")) {
					data = LoadARFF.loadARFF(Constants.ARFF_NUMERIC_PATH);
				}
				data.setClassIndex(data.numAttributes() - 1); // Set play {yes, no}
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
				data.setClassIndex(data.numAttributes() - 1); // Set play {yes, no}
				break;
			case "3":
				/** Remove attribute (outlook) */
				Enumeration<Attribute> e;
				
				e = data.enumerateAttributes();
				TextWriter.printEnumerationAttribute(e);

				data.deleteAttributeAt(0); // Delete first attribute - Outlook
				
				e = data.enumerateAttributes();
				TextWriter.printEnumerationAttribute(e);
				break;
			case "4":
				/** Filter (resample) */
				System.out.println("# Previous : " + data.numInstances());
				data = SupervisedFilter.resampleInstances(data);
				System.out.println("# After : " + data.numInstances());
				break;
			case "5":
				/** Build classifier with Naive Bayes */
				TextWriter.printClassifierMenu();
				input2 = reader.readLine();
				if (input2.equals("1")) {
					cModel = ClassifyAlgorithm.naiveBayesAlgorithm(data, 1);
				} else if (input2.equals("2")) {
					cModel = ClassifyAlgorithm.naiveBayesAlgorithm(data, 2);
				}
				break;
			case "6":
				/** Build classifier with DT */
				TextWriter.printClassifierMenu();
				input2 = reader.readLine();
				if (input2.equals("1")) {
					cModel = ClassifyAlgorithm.iD3Algorithm(data, 1);
				} else if (input2.equals("2")) {
					cModel = ClassifyAlgorithm.iD3Algorithm(data, 2);
				}
				break;
			case "7":
				/** Testing model given test set */
				
				break;
			case "8":
				/** Testing model to classify one unseen data */
				
				break;
			case "9":
				/** Save model */
				
				break;
			case "10":
				/** Load model */
				
				break;
			case "11":
				/** Classify using extended classifier */
				
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
