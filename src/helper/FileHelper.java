package helper;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

import weka.classifiers.Classifier;

/**
 * File Helper for Application
 * 
 * @author Iskandar Setiadi
 * @version 0.1, by IS @since September 16, 2014
 *
 */

public class FileHelper {

	/**
	 * Save cModel to path
	 * @param cModel
	 * @param path
	 * @throws Exception
	 */
	public static void saveModel(Classifier cModel, String path)
			throws Exception {
		ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(
				path));
		oos.writeObject(cModel);
		oos.flush();
		oos.close();
	}

	/**
	 * Load cModel from path
	 * @param path
	 * @return
	 * @throws Exception
	 */
	public static Classifier loadModel(String path) throws Exception {
		ObjectInputStream ois = new ObjectInputStream(new FileInputStream(path));
		Classifier cModel = (Classifier) ois.readObject();
		ois.close();
		return cModel;
	}

}
