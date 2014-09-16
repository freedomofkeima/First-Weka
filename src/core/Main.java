package core;

import helper.Constants;

import java.io.BufferedReader;
import java.io.InputStreamReader;

import loader.LoadARFF;
import loader.LoadCSV;
import weka.core.Instances;

/**
 * Main class for Weka
 * 
 * @author Iskandar Setiadi
 * @version 0.1, by IS @since September 16, 2014
 *
 */

public class Main {

	public static void main(String[] args) throws Exception {

		Instances data = null;

		BufferedReader reader = new BufferedReader(new InputStreamReader(
				System.in));
		while (true) {
			/** Menu settings */
			System.out.println("-- Menu -- ");
			System.out.println("1 - Test Load from .arff (Nominal)");
			System.out.println("2 - Test Load from .arff (Numeric)");
			System.out.println("3 - Test Load from .csv");
			System.out.println("999 - Exit");
			System.out.print("Input: ");
			/** Input */
			String input = reader.readLine();
			switch (input) {
			case "1":
				/** Case 1 logic here */
				data = LoadARFF.loadARFF(Constants.ARFF_NOMINAL_PATH);
				break;
			case "2":
				/** Case 2 logic here */
				data = LoadARFF.loadARFF(Constants.ARFF_NUMERIC_PATH);
				break;
			case "3":
				/** Case 3 logic here */
				data = LoadCSV.loadCSV(Constants.CSV_PATH);
				break;
			case "999":
				return;
			default:
				System.out.println("Unrecognized input value!");
			}
		}

	}

}
