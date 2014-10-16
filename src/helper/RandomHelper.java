package helper;

import java.util.Random;

/**
 * Randomize Helper
 * 
 * @author WbTeladan
 * @version 0.1, by WbTeladan @since October 15, 2014
 *
 */

public class RandomHelper {
	
	private static double RANDOM_INTERVAL = 1; // from -1 to 1
	private static Random generator = new Random(System.currentTimeMillis());
	
	
	/** Random tool */
	public static double randomValue() {
		double num = generator.nextDouble() * RANDOM_INTERVAL * 2;
		num -= RANDOM_INTERVAL;
		return num;
	}

}
