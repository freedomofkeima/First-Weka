package clusterer;

import weka.clusterers.ClusterEvaluation;
import weka.clusterers.Clusterer;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.CapabilitiesHandler;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.RevisionUtils;
import weka.core.SerializedObject;
import weka.core.Utils;

/**
 * Custom Hierarchical Clustering Implementation (implements Clusterer)
 * 
 * @author WbTeladan
 * @version 0.1, by WbTeladan @since December 4, 2014
 *
 */

public class myHierarchicalClusterer implements Clusterer, CapabilitiesHandler {
	
	/** List of Attributes */
	private int nCluster = 2; // number of cluster

	public void buildClusterer(Instances data) throws Exception {

	}

	public int clusterInstance(Instance instance) throws Exception {

		double[] dist = distributionForInstance(instance);

		if (dist == null) {
			throw new Exception("Null distribution predicted");
		}

		if (Utils.sum(dist) <= 0) {
			throw new Exception("Unable to cluster instance");
		}
		return Utils.maxIndex(dist);
	}

	public double[] distributionForInstance(Instance instance) throws Exception {

		double[] d = new double[numberOfClusters()];

		d[clusterInstance(instance)] = 1.0;

		return d;
	}

	public int numberOfClusters() throws Exception {
		return 0;
	}

	public static Clusterer forName(String clustererName, String[] options)
			throws Exception {
		return (Clusterer) Utils.forName(Clusterer.class, clustererName,
				options);
	}

	public static Clusterer makeCopy(Clusterer model) throws Exception {
		return (Clusterer) new SerializedObject(model).getObject();
	}

	public static Clusterer[] makeCopies(Clusterer model, int num)
			throws Exception {
		if (model == null) {
			throw new Exception("No model clusterer set");
		}
		Clusterer[] clusterers = new Clusterer[num];
		SerializedObject so = new SerializedObject(model);
		for (int i = 0; i < clusterers.length; i++) {
			clusterers[i] = (Clusterer) so.getObject();
		}
		return clusterers;
	}

	public String getRevision() {
		return RevisionUtils.extract("$Revision: 5537 $");
	}

	protected static void runClusterer(Clusterer clusterer, String[] options) {
		try {
			System.out.println(ClusterEvaluation.evaluateClusterer(clusterer,
					options));
		} catch (Exception e) {
			if ((e.getMessage() == null)
					|| ((e.getMessage() != null) && (e.getMessage().indexOf(
							"General options") == -1)))
				e.printStackTrace();
			else
				System.err.println(e.getMessage());
		}
	}

	@Override
	public Capabilities getCapabilities() {
		Capabilities result = new Capabilities(this);
		result.disableAll();
		result.enable(Capability.NO_CLASS);

		result.enable(Capability.NOMINAL_ATTRIBUTES);
		result.enable(Capability.NUMERIC_ATTRIBUTES);
		result.enable(Capability.DATE_ATTRIBUTES);
		result.enable(Capability.MISSING_VALUES);
		result.enable(Capability.STRING_ATTRIBUTES);

		result.setMinimumNumberInstances(0);
		return result;
	}

	public int getnCluster() {
		return nCluster;
	}

	public void setnCluster(int nCluster) {
		this.nCluster = nCluster;
	}

}
