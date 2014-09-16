package filter;

import weka.core.Instances;
import weka.filters.supervised.instance.Resample;

/**
 * Supervised Filter for Application 
 * 
 * @author Iskandar Setiadi
 * @version 0.1, by IS @since September 16, 2014
 *
 */

public class SupervisedFilter {
	
	/**
	 * Resample Instances (for test purposes, we take 75% size)
	 * @param i
	 * @return
	 * @throws Exception
	 */
	public static Instances resampleInstances(Instances i) throws Exception {
		String Filteroptions="-B 1.0";
		Resample sampler = new Resample();
		/** Resample Options */
		sampler.setOptions(weka.core.Utils.splitOptions(Filteroptions));
		sampler.setRandomSeed((int)System.currentTimeMillis());
		sampler.setSampleSizePercent(75.0);
		sampler.setInputFormat(i);
		
		i = Resample.useFilter(i, sampler);
		
		return i;
	}

}
