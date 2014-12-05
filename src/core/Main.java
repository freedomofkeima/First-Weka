package core;

import filter.SupervisedFilter;
import helper.Constants;
import helper.FileHelper;
import helper.TextWriter;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.Enumeration;
import java.util.Random;

import loader.LoadARFF;
import loader.LoadCSV;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.Id3;
import weka.clusterers.ClusterEvaluation;
import weka.clusterers.Clusterer;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import classifier.ClassifyAlgorithm;
import classifier.CustomAlgorithm;
import classifier.NeuronLayer;
import classifier.myID3;
import classifier.myNN;
import clusterer.myHierarchicalClusterer;
import clusterer.myPartitionalClusterer;

/**
 * Main class for Weka
 * 
 * @author Iskandar Setiadi
 * @version 0.1, by IS @since September 16, 2014
 *
 */

public class Main {

	@SuppressWarnings({ "unchecked", "rawtypes" })
	public static void main(String[] args) throws Exception {

		Instances data = null;
		Classifier cModel = null;
		String input, input2;
		boolean isNominal = true;

		BufferedReader reader = new BufferedReader(new InputStreamReader(
				System.in));
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
					isNominal = false;
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
					isNominal = false;
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
					Instance test = new Instance(5);
					if (isNominal) {
						test.setValue(data.attribute(0), "sunny");
						test.setValue(data.attribute(1), "mild");
						test.setValue(data.attribute(2), "high");
						test.setValue(data.attribute(3), "FALSE");
					} else {
						test.setValue(data.attribute(0), "rainy");
						test.setValue(data.attribute(1), 65);
						test.setValue(data.attribute(2), 70);
						test.setValue(data.attribute(3), "TRUE");
					}
					// Give access to dataset
					test.setDataset(data);

					System.out.println("Classifying result:");
					System.out.println(data.attribute(data.numAttributes() - 1)
							.value((int) cModel.classifyInstance(test)));
				} else {
					System.out.println("You need to build classifier first!");
				}
				break;
			case "9":
				/** Save model */
				if (cModel != null) {
					FileHelper.saveModel(cModel, Constants.SAVE_MODEL_PATH);
				} else {
					System.out.println("You need to build classifier first!");
				}
				break;
			case "10":
				/** Load model */
				cModel = FileHelper.loadModel(Constants.SAVE_MODEL_PATH);
				break;
			case "11":
				/** Create an extended classifier */
				if (data != null) {
					cModel = new CustomAlgorithm();
					cModel.buildClassifier(data);

					// Test to classify data1
					Evaluation eval = new Evaluation(data);
					eval.evaluateModel(cModel, data);
					System.out.println(eval.toSummaryString(
							"\nResults\n======\n", false));
				} else {
					System.out.println("You need to load your data first!");
				}
				break;
			case "12":
				/** Custom ID3 implementation */
				if (data != null) {
					cModel = new myID3();
					/** 10 Cross-fold */
					Evaluation eval = new Evaluation(data);
					eval.crossValidateModel(cModel, data, 10, new Random(1));
					System.out.println(eval.toSummaryString(
							"\nResults\n======\n", false));
					cModel.buildClassifier(data);
					System.out.println(cModel.toString());
					/** End of Building Model section */
				} else {
					System.out.println("You need to load your data first!");
				}
				break;
			case "13":
			case "14":
				if (data != null) {
					Instances data_complement = LoadARFF
							.loadARFF(Constants.ARFF_NOMINAL_COMPLEMENT_PATH);

					if (input.equals("13")) {
						cModel = new myID3();
					} else {
						cModel = new Id3();
					}

					Evaluation eval = new Evaluation(data);
					cModel.buildClassifier(data);
					System.out.println(cModel.toString());

					/** Classify dataset complement */
					Enumeration instEnum = data_complement.enumerateInstances();
					while (instEnum.hasMoreElements()) {
						Instance inst = (Instance) instEnum.nextElement();
						inst.setDataset(data);
						System.out.println(inst.toString());
						System.out.println("Classifying result:");
						double value = cModel.classifyInstance(inst);
						inst.setClassValue(value);
						System.out.println(data.attribute(
								data.numAttributes() - 1).value((int) value));
						System.out.println();
					}

					/** Re-evaluate back */
					System.out.println("Re-evaluate dataset 3 with model 1:");
					eval.evaluateModel(cModel, data_complement);
					System.out.println();
					System.out.println(eval.toSummaryString(
							"\nResults\n======\n", false));

					Instances data_noise = LoadARFF
							.loadARFF(Constants.ARFF_NOMINAL_NOISE_PATH);
					Classifier cModel_noise;

					if (input.equals("13")) {
						cModel_noise = new myID3();
					} else {
						cModel_noise = new Id3();
					}

					// Set play {yes, no}
					data_noise.setClassIndex(data_noise.numAttributes() - 1);

					Evaluation eval_noise = new Evaluation(data_noise);
					cModel_noise.buildClassifier(data_noise);
					System.out.println(cModel_noise.toString());

					/** Evaluate dataset 3 with model 2 */
					System.out.println("Evaluate dataset 3 with model 2:");
					eval_noise.evaluateModel(cModel_noise, data_complement);
					System.out.println();
					System.out.println(eval_noise.toSummaryString(
							"\nResults\n======\n", false));

				} else {
					System.out.println("You need to load your data first!");
				}
				break;
			case "15":
				// choose NN Type
				TextWriter.printNNMenu();
				input2 = reader.readLine();
				
				// load data
				data = LoadARFF.loadARFF(Constants.ARFF_DATASET2_PATH);
				data.setClassIndex(data.numAttributes() - 1);
				
				// create model
				myNN model = null;
				
				/**
				 *  Define custom test layer here 
				 *  Number of input nodes = number of features (without class index)
				 *  Available parameters:
				 *  - learning_rate
				 *  - momentum 
				 *  - max_epoch
				 *  - min_error (MSE)
				 *  - threshold (if > threshold then "TRUE" else "FALSE", unary output)
				 *  - activation_type
				 * 
				 **/
				NeuronLayer i_layer;
				NeuronLayer o_layer;
				switch(input2) {
				case "1":
				case "2":
				case "3":
					model = new myNN(Integer.parseInt(input2));
					i_layer = new NeuronLayer(model.getLayerSize(), 2 + 1); // 1 phantom
					model.addLayer(i_layer);
					o_layer = new NeuronLayer(model.getLayerSize(), 1);
					// o_layer.getNodes(0).setWeight(0, 0.5);
					// o_layer.getNodes(0).setWeight(1, 0.5);
					// o_layer.getNodes(0).setWeight(2, 0.5);
					model.addLayer(o_layer);
					
					/** 
					 * Customize (Given) example
					 * model.getLayers(1).getNodes(0).setWeight(0, 0.5);
					 */
					
					// hardlims activation function
					if (input2.equals("1")) {
						model.getLayers(1).getNodes(0).setActivation_type(3);
					}
					else model.getLayers(1).getNodes(0).setActivation_type(4);
					// learning rate should be small
					model.setLearning_rate(0.4);
					// maximum number of epoch
					model.setMax_epoch(500);
					// MSE configuration
					model.setMin_error(0.001);
					
					model.buildClassifier(data);
					break;
				case "4":
					model = new myNN(4);
					i_layer = new NeuronLayer(model.getLayerSize(), data.numAttributes() - 1);
					model.addLayer(i_layer);
					NeuronLayer hidden_layer = new NeuronLayer(model.getLayerSize(), 8);
					model.addLayer(hidden_layer);
					o_layer = new NeuronLayer(model.getLayerSize(), 1);
					model.addLayer(o_layer);

					// sigmoid threshold ( 0..1 )
					// model.setThreshold(0.5); // for binary classification
					// learning rate should be small
					model.setLearning_rate(0.1);
					// momentum configuration
					model.setMomentum(0.7);
					// maximum number of epoch
					model.setMax_epoch(500);
					// MSE configuration
					model.setMin_error(0.05);
					
					model.buildClassifier(data);

					break;
				default:
					System.out.println("Unrecognized input value!");
				}
				
				Evaluation eval = new Evaluation(data);
				eval.evaluateModel(model, data);
				System.out.println(eval.toSummaryString(
						"\nResults\n======\n", false));
				
				
				/*
				// Classifying new instance
				Instance test = new Instance(3);
				test.setValue(data.attribute(0), 1);
				test.setValue(data.attribute(1), 0);
				System.out.println("\nTest data:");
				System.out.println(test.value(0) + " " + test.value(1));
				// Give access to dataset
				test.setDataset(data);
				System.out.println("Classifying result:");
				System.out.println((int) model.classifyInstance(test));
				*/
				break;
			case "16":
				if (data != null) {
					Clusterer clusterer = new myHierarchicalClusterer(1); // single_link
					clusterer.buildClusterer(data);
					
					/**
					 * Notice: clusterInstance() to choose the cluster index
					 */
					ClusterEvaluation cluster_eval = new ClusterEvaluation();
					cluster_eval.setClusterer(clusterer);
					cluster_eval.evaluateClusterer(data);
					System.out.println("\nResults\n======\n");
					System.out.println(cluster_eval.clusterResultsToString());
				}
				break;
			case "17":

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
