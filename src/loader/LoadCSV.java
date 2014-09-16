package loader;

import java.io.File;

import weka.core.Instances;
import weka.core.converters.CSVLoader;

/**
 * CSV Loader for Application This file is adapted from
 * /wekaexamples/core/converters/LoadDataFromCsvFile.java
 * 
 * @author Iskandar Setiadi
 * @version 0.1, by IS @since September 16, 2014
 *
 */

public class LoadCSV {

	public static Instances loadCSV(String path) throws Exception {
		Instances data = null;

		System.out.println("\nReading file " + path + "...");
		CSVLoader loader = new CSVLoader();
		loader.setSource(new File(path));
		data = loader.getDataSet();

		System.out.println("\nHeader of dataset:\n");
		System.out.println(new Instances(data, 0));

		return data;
	}

}
