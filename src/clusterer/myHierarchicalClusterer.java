package clusterer;

import java.util.ArrayList;
import java.util.Enumeration;

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
	private int n_cluster = 2; // number of cluster
	/**
	 * SINGLE = 1 COMPLETE = 2
	 */
	private final int link_type;
	private Instances instances; // instances
	private int current_n_cluster; // current number of cluster -> end when this
									// value == 1

	class Cluster {
		Cluster left;
		Cluster right;
		double left_distance;
		double right_distance;
		ArrayList<Instance> elements;
		int height;

		Cluster() {
			elements = new ArrayList<Instance>();
		}

		Cluster get_left() {
			return left;
		}

		Cluster get_right() {
			return right;
		}

		void set_left(Cluster left, double distance) {
			this.left = left;
			left_distance = distance;
		}

		void set_right(Cluster right, double distance) {
			this.right = right;
			right_distance = distance;
		}

		Instance get_element(int idx) {
			return elements.get(idx);
		}

		void add_element(Instance i) {
			elements.add(i);
		}

		void set_element(Instance i, int idx) {
			elements.set(idx, i);
		}

		void set_elements(ArrayList<Instance> i) {
			elements = i;
		}

		void set_distance(double d1, double d2) {
			left_distance = d1;
			right_distance = d2;
		}
		int get_height() {
			return height;
		}
		void set_height(int height) {
			this.height = height;
		}
	}

	private Cluster[] clusters;

	public myHierarchicalClusterer(int link_type) {
		this.link_type = link_type;
	}

	@SuppressWarnings("unchecked")
	public void buildClusterer(Instances data) throws Exception {
		instances = data;
		if (instances.numInstances() == 0)
			return;
		current_n_cluster = instances.numInstances();
		
		if (n_cluster > current_n_cluster) n_cluster = current_n_cluster;
		
		clusters = new Cluster[current_n_cluster];

		/** Initial: Each instances -> 1 cluster */
		@SuppressWarnings("rawtypes")
		Enumeration e = instances.enumerateInstances();
		int idx = 0;
		while (e.hasMoreElements()) {
			Instance inst = (Instance) e.nextElement();
			Cluster entity = new Cluster();
			entity.add_element(inst); // add element
			entity.set_height(0); // leaf, default height
			clusters[idx] = entity;
			idx++; // iterate to next element
		}
		
		/** Process: Link each cluster until single cluster
		 *  Notice: Just do dirty hack, copy all elements from leaf to current cluster node
		 */
		while (current_n_cluster != 1) {
			
		}
		
		
		/**
		 * Finalize: Equalize current_n_cluster to n_cluster
		 */
		
		
	}
	
	/** Search for two clusters to be merged */
	private void linkCluster() {
		/** Compute distance between two clusters */
		// TODO: Optimize with DP / Pre-compute
		
		/** Merge two clusters with minimum distance (single link) or maximum distance (complete link) */
		
	}
	
	/**
	 * Use EuclideanDistance to get distance between two clusters
	 */
	private double getDistance(Cluster c1, Cluster c2) {
		switch(link_type) {
		case 1: // single link
			break;
		case 2: // complete link
			break;
		}
		return 0;
	}

	public int clusterInstance(Instance instance) throws Exception {
		if (instances.numInstances() == 0) {
			return 0;
		}
		/** 
		 * Search for nearest instance (model) which represents current `instance`
		 * Return cluster index of the instance (model)
		 */
		for (int i = 0; i < instances.numInstances(); i++) {
			
		}
		/** DUMMY */
		return 0;
	}

	public double[] distributionForInstance(Instance instance) throws Exception {
		if (numberOfClusters() == 0) {
			double[] p = new double[1];
			p[0] = 1;
			return p;
		}
		double[] p = new double[numberOfClusters()];
		p[clusterInstance(instance)] = 1.0;
		return p;
	}

	public int numberOfClusters() throws Exception {
		return n_cluster;
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

	public int getn_cluster() {
		return n_cluster;
	}

	public void setnCluster(int n_cluster) {
		this.n_cluster = n_cluster;
	}

}
