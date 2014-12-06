package clusterer;

import java.util.ArrayList;
import java.util.Enumeration;

import weka.clusterers.ClusterEvaluation;
import weka.clusterers.Clusterer;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.CapabilitiesHandler;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.RevisionUtils;
import weka.core.SerializedObject;
import weka.core.Utils;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.RemoveFrequentValues;

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

	class Cluster implements Cloneable {
		Cluster left;
		Cluster right;
		ArrayList<Instance> elements;
		ArrayList<Instance> lastelements;//simpen instances sebelum nya , kalo yg baru sama dengan yg lama algo selesai
		Instance core;
		double epsilon=0.01;//
		int height;

		Cluster() {
			elements = new ArrayList<Instance>();
			lastelements = new ArrayList<Instance>();
		}

		Cluster getClone() {
			try {
				// call clone
				return (Cluster) super.clone();
			} catch (CloneNotSupportedException e) {
				System.out.println(" Cloning not allowed. ");
				return this;
			}
		}

		Cluster get_left() {
			return left;
		}

		Cluster get_right() {
			return right;
		}

		void set_left(Cluster left) {
			this.left = left;
		}

		void set_right(Cluster right) {
			this.right = right;
		}

		Instance get_element(int idx) {
			return elements.get(idx);
		}

		int get_element_size() {
			return elements.size();
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

		int get_height() {
			return height;
		}

		void set_height(int height) {
			this.height = height;
		}
		
		//emon
		void set_core(){
			int[] count = new int[elements.get(0).numAttributes()];
			ArrayList<ArrayList<Integer>> group = new ArrayList<ArrayList<Integer>>(elements.get(0).numAttributes());
			lastelements.clear();
			for(int i=0;i<elements.size();i++){
				lastelements.add(elements.get(i));
				for (int j =0;j<elements.get(i).numAttributes();j++){
					if(!elements.get(i).attribute(j).isNumeric()){
						
					}else{//averaging
						count[j]+= elements.get(i).value(j);
					}
					if(i == elements.size()-1){
						if(elements.get(i).attribute(j).isNumeric())
							core.setValue(j, count[j]/elements.size());
						//else //ini yang datanya musti count
					}
				}
			}
		}
		
		int getDistance(Instance i){
			int count = 0;
			for(int j=0;j<core.numAttributes();j++){
				if(core.attribute(j).isNumeric()){
					if(Math.abs(i.value(j) - core.value(j)) < epsilon)//kalo lebih besar dari batas toleransi, jaraknya naik
						count++;
				}else{ //sama atau kagak
					if (i.stringValue(j).equals(core.stringValue(j)))
						count++;
				}
			}
		return count;
		}
		
		boolean isConvergen(){
			boolean convergen = false;
				if(elements.size()!=lastelements.size())
					convergen = false;
				else{//bandingin attribute2 di dalemnya
					
				}
			return convergen;
		}
		
	}

	private ArrayList<Cluster> clusters;

	public myHierarchicalClusterer(int link_type) {
		this.link_type = link_type;
	}

	public void buildClusterer(Instances data) throws Exception {
		instances = data;
		if (instances.numInstances() == 0)
			return;
		current_n_cluster = instances.numInstances();

		if (n_cluster > current_n_cluster)
			n_cluster = current_n_cluster;

		clusters = new ArrayList<Cluster>();

		/** Initial: Each instances -> 1 cluster */
		@SuppressWarnings("rawtypes")
		Enumeration e = instances.enumerateInstances();
		while (e.hasMoreElements()) {
			Instance inst = (Instance) e.nextElement();
			Cluster entity = new Cluster();
			entity.add_element(inst); // add element
			entity.set_height(0); // leaf, default height
			clusters.add(entity);
		}

		/**
		 * Process: Link each cluster until single cluster Notice: Just do dirty
		 * hack, copy all elements from leaf to current cluster node
		 */
		while (current_n_cluster != 1) {
			/** Compute distance between two clusters */
			// TODO (@freedomofkeima): Optimize with DP / Pre-compute

			double[][] distance = new double[current_n_cluster][current_n_cluster];
			// Initialize
			for (int i = 0; i < current_n_cluster; i++)
				for (int j = 0; j < current_n_cluster; j++)
					distance[i][j] = -1;
			for (int i = 0; i < current_n_cluster; i++)
				for (int j = 0; j < current_n_cluster; j++)
					if (i != j && distance[j][i] == -1) {
						distance[i][j] = getDistance(clusters.get(i),
								clusters.get(j));
						distance[j][i] = distance[i][j]; // symmetric
					}
			// Search for minimum / maximum from all pairs
			int idx_one = 0, idx_two = 1; // initialize
			double dist = distance[idx_one][idx_two];
			for (int i = 0; i < current_n_cluster; i++)
				for (int j = 0; j < current_n_cluster; j++) {
					if (i != j) {
						switch (link_type) {
						case 1:
							if (dist > distance[i][j]) { // symmetry,
															// interchangeable
								idx_one = i;
								idx_two = j;
							}
							break;
						case 2:
							if (dist < distance[i][j]) { // symmetry,
															// interchangeable
								idx_one = i;
								idx_two = j;
							}
							break;
						}
					}
				}
			// Link
			Cluster new_cluster = linkCluster(clusters.get(idx_one),
					clusters.get(idx_two));
			clusters.set(idx_one, new_cluster);
			clusters.remove(idx_two);
			current_n_cluster--;
		}

		/**
		 * Finalize: Equalize current_n_cluster to n_cluster
		 */
		System.out.println(clusters.get(0).get_element_size());
		System.out.println(clusters.get(0).get_height());
		System.out.println("Left: " + clusters.get(0).get_left().get_element_size());
		System.out.println(clusters.get(0).get_left().get_height());
		System.out.println("Right: " + clusters.get(0).get_right().get_element_size());
		System.out.println(clusters.get(0).get_right().get_height());
		// TODO (@hotarufk)
		
	}

	/** Merge two clusters */
	private Cluster linkCluster(Cluster c1, Cluster c2) {
		/**
		 * Merge two clusters with minimum distance (single link) or maximum
		 * distance (complete link)
		 */
		Cluster result = new Cluster();
		result.set_height(Math.max(c1.get_height(), c2.get_height()) + 1);

		// Set left & right
		result.set_left(c1.getClone());
		result.set_right(c2.getClone());

		for (int i = 0; i < c1.get_element_size(); i++)
			result.add_element(c1.get_element(i));
		for (int i = 0; i < c2.get_element_size(); i++)
			result.add_element(c2.get_element(i));

		return result;
	}

	/**
	 * Use EuclideanDistance to get distance between two clusters
	 */
	private double getDistance(Cluster c1, Cluster c2) {
		double global_dist = 0;
		for (int i = 0; i < c1.get_element_size(); i++)
			for (int j = 0; j < c2.get_element_size(); j++) {
				global_dist += getDistanceInstance(c1.get_element(i), c2.get_element(j));
			}

		return global_dist;
	}
	
	private double getDistanceInstance(Instance i1, Instance i2) {
		double temp_dist = 0;
		for (int k = 0; k < instances.numAttributes() - 1; k++) {
			if (instances.attribute(k).isNominal()) {
				if (Math.abs(i1.value(k) - i2.value(k)) > 0.01)
					temp_dist = temp_dist + 1;
			}
			if (instances.attribute(k).isNumeric()) {
				// TODO @hotarufk: Check bound lower..upper for each attribute (not hardcoded)
				double lower_bound = 64;
				double upper_bound = 96;
				
				if (Math.abs(lower_bound - upper_bound) > 0.000001) {
					double diff = (i1.value(k) - i2.value(k))
							/ (upper_bound - lower_bound);
					temp_dist = temp_dist + diff * diff;
				}
			}
		}
		return Math.sqrt(temp_dist);
	}

	@SuppressWarnings("rawtypes")
	public int clusterInstance(Instance instance) throws Exception {
		if (instances.numInstances() == 0) {
			return 0;
		}
		/**
		 * Search for nearest instance (model) which represents current
		 * `instance` Return cluster index of the instance (model)
		 */
		int idx = -1, counter = 0;
		double dist = 1000000000;
		Enumeration e = instances.enumerateInstances();
		while (e.hasMoreElements()) {
			Instance inst = (Instance) e.nextElement();
			double distance = getDistanceInstance(inst, instance);
			if (idx == -1) {
				idx = counter;
				dist = distance;
			} else if (dist > distance) {
				idx = counter;
				dist = distance;
			}
			counter++;
		}
		
		// Search for the k-th Cluster of Instances.get(idx)
		// Return k
		// TODO (@hotarufk)
		
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
