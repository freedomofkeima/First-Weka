package clusterer;

import clusterer.myHierarchicalClusterer.Cluster;
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
 * Custom Partitional Clustering Implementation (implements Clusterer)
 * 
 * @author WbTeladan
 * @version 0.1, by WbTeladan @since December 4, 2014
 *
 */

public class myPartitionalClusterer implements Clusterer, CapabilitiesHandler  {
//cara kerja k-means
//TENTUIN BAKAL ADA BERAPA CLUSTER
//Randomize Center nya yg mana aja -> attribut center nyolong dari data yg di randomize
//sisa data diclustering ke center2 yg terdekat -> buat algoritma comparision nya
//kalo udah semua di clustering, hitung ulang centernya -> pake fungsi ngitung center berdasarkan data yang ada -> sama kayak fungsi awal pas randomize
//hitung jarak isi cluster ke center sendiri dan center orang lain, kalo center orang lain lebih deket -> pindahin ke sana
//ulangi dari tahap 2
//tresshold kalo gak ada yg berubah2 clusternya atau udh x iterasi(maunya)


	/** List of Attributes */
	private int nCluster = 2; // number of cluster
	private Cluster[] CC;
	private Instances data;
	private int Max_Itrations;
	
	//private Instance[] clustercore;

	public void buildClusterer(Instances data) throws Exception {
		this.data = data;
		CC = new Cluster[nCluster];
		//masukin core awal ke CC
		for(int i=0;;i++){
			CC[i].add_element(this.data.instance(1));
			//hitung bagian tengah dari core
		}
	//	clustercore = new Instance[nCluster];
	}

	public int clusterInstance(Instance instance) throws Exception { //fungsi buat ngitung jaraknnya

		double[] dist = distributionForInstance(instance);

		if (dist == null) {
			throw new Exception("Null distribution predicted");
		}

		if (Utils.sum(dist) <= 0) {
			throw new Exception("Unable to cluster instance");
		}
		return Utils.minIndex(dist);
	}

	public double[] distributionForInstance(Instance instance) throws Exception {//fungsi buat ngitung jaraknnya
		double[] d = new double[numberOfClusters()];
		//hitung jarak ke cluster
		for(int i=0;i<nCluster;i++)
			d[i]=this.CC[i].getDistance(instance);	
		return d;
	}
	

	public int numberOfClusters() throws Exception {
		return nCluster;
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
