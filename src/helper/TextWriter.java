package helper;

import java.util.Enumeration;

import weka.core.Attribute;

/**
 * Text Writer for Application
 * 
 * @author Iskandar Setiadi
 * @version 0.1, by IS @since September 16, 2014
 *
 */

public class TextWriter {
	
	/**
	 * Print Main Menu
	 */
	public static void printMainMenu() {
		/** Menu settings */
		System.out.println("-- Menu -- ");
		System.out.println("1 - Test load from .arff");
		System.out.println("2 - Test load from .csv");
		System.out.println("3 - Test remove attribute (outlook)");
		System.out.println("4 - Filter (resample)");
		/** 
		 * In no 5 & 6, you can choose between 10-fold cross validation 
		 * or percentage split 
		 * */
		System.out.println("5 - Build classifier with Naive Bayes");
		System.out.println("6 - Build classifier with DT");
		System.out.println("7 - Testing model given test set");
		System.out.println("8 - Testing model to classify one unseen data");
		System.out.println("9 - Save model");
		System.out.println("10 - Load model");
		System.out.println("11 - Create an extended Classifier");
		System.out.println("12 - Using Custom ID3 implementation");
		System.out.println("13 - Praktikum 1 (myID3)");
		System.out.println("14 - Praktikum 1 (ID3 Weka)");
		System.out.println("15 - Praktikum 2 (myNN)");
		System.out.println("16 - Clustering with Hierarchical");
		System.out.println("17 - Clustering with Partitional");
		System.out.println("999 - Exit");
		System.out.print("Input: ");
	}
	
	/**
	 * Print Load Menu
	 */
	public static void printLoadMenu() {
		System.out.println("-- Load --");
		System.out.println("1 - Nominal data");
		System.out.println("2 - Numeric data");
		System.out.print("Input: ");
	}
	
	/**
	 * Print Classifier Menu
	 */
	public static void printClassifierMenu() {
		System.out.println("-- Classifier --");
		System.out.println("1 - 10-fold cross-validation");
		System.out.println("2 - Percentage Split (50%)");
		System.out.print("Input: ");
	}
	
	/**
	 * Enumerate through Attributes
	 * @param e
	 */
	public static void printEnumerationAttribute(Enumeration<Attribute> e) {
		System.out.println("-- List of Attributes --");
		while (e.hasMoreElements()) {
			Attribute element = e.nextElement();
			System.out.println(element);
		}
	}
	
	/**
	 * Print NN Menu
	 * Required Parameter:
	 * - Initial Weight (random / given)
	 * - Fungsi Aktivasi per Neuron
	 * - Learning Rate
	 * - Momentum
	 * - Terminasi (deltaMSE, maxIteration)
	 * 
	 */
	public static void printNNMenu() {
		System.out.println("-- MyNN --");
		System.out.println("1 - Single Perceptron");
		System.out.println("2 - Batch Gradient Descent");
		System.out.println("3 - Delta Rule");
		System.out.println("4 - Back Propagation (MLP)");
		System.out.print("Input: ");
	}

}
