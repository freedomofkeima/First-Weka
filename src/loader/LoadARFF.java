package loader;

import java.io.File;

import weka.core.Instances;
import weka.core.converters.ArffLoader;

/**
 * ARFF Loader for Application
 * This file is adapted from 
 * /wekaexamples/core/converters/LoadDataFromArffFile.java
 * 
 * @author Iskandar Setiadi
 * @version 0.1, by IS @since September 16, 2014
 *
 */

public class LoadARFF {
	
	public static Instances loadARFF(String path) throws Exception {
		Instances data = null;
		
	    System.out.println("\nReading file " + path + "...");
	    ArffLoader loader = new ArffLoader();
	    if (path.startsWith("http:") || path.startsWith("ftp:"))
	      loader.setURL(path);
	    else
	    // loader.setSource(new File(path));
	    	 loader.setSource(new File("D:/Andri/Kuliah/ML/Weka/data/weather.numeric.arff"));
	    data = loader.getDataSet();
	    
	    System.out.println("\nHeader of dataset:\n");
	    System.out.println(new Instances(data, 0));
		
		return data;
	}

}
