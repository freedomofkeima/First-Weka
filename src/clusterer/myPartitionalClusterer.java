package clusterer;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;

import clusterer.Cluster;
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
	private ArrayList<Cluster> CC = new ArrayList<Cluster>();
	private Instances data;
	private static int Max_Itrations = 1000;
	
	//private Instance[] clustercore;

	public myPartitionalClusterer(int nCluster) {
		this.nCluster = nCluster;
		
	}
	
	public void buildClusterer(Instances newdata) throws Exception {
		data = new Instances(newdata);
		data.setClassIndex(-1);
		int clusternumber =0;
		//masukin core awal ke CC
		Random r = new Random();
		Set set = new HashSet();
		while (set.size() <= nCluster) 
		    set.add(r.nextInt(data.numInstances()));
		Object[]hasil = set.toArray();
		//System.out.println("isi set"+Integer.parseInt(hasil[0].toString()));
		//System.out.println("isi set"+hasil[1].toString());
		for(int i=0;i<nCluster;i++){
			Cluster C = new Cluster(data);
			CC.add(C);
			CC.get(i).add_element(data.instance(1));
			CC.get(i).set_centroid();
		}
		for(int i=0;i<Max_Itrations;i++){
			for(int j=0;j<nCluster;j++)
				CC.get(j).elements.clear();
			for(int j=0;j<data.numInstances();j++){

				CC.get(clusternumber).add_element(data.instance(j));
				if(j== data.numInstances()-1){//hitung ulang corenya
					boolean isConvergen = false;
					for(int k=0;k<nCluster;k++)
						isConvergen = CC.get(k).isConvergen();
					if(isConvergen)
						break;
					else{ //data last dengan yg sekarang beda
						for(int k=0;k<nCluster;k++)
							CC.get(k).set_centroid();
					}
				}
			}
		}
	//	clustercore = new Instance[nCluster];
	}

	public int clusterInstance(Instance instance) throws Exception { //fungsi buat ngitung jaraknnya

		double[] score = distributionForInstance(instance);

		if (score == null) {
			throw new Exception("Null distribution predicted");
		}

		if (Utils.sum(score) <= 0) {
			throw new Exception("Unable to cluster instance");
		}
		return Utils.maxIndex(score);
	}

	public double[] distributionForInstance(Instance instance) throws Exception {//fungsi buat ngitung jaraknnya
		double[] d = new double[numberOfClusters()];
		//hitung jarak ke cluster
		
		for(int i=0;i<nCluster;i++){
			instance.setDataset(CC.get(0).test);
			d[i]=distance(instance,CC.get(i).centroid);
		}
		return d;
	}
	
	public int distance(Instance first,Instance second){
		int distance = 0;
		second.setDataset(data);
		for(int j=0;j<first.numAttributes();j++){
			if(second.attribute(j).isNumeric()){
				if(Math.abs(first.value(j) - second.value(j)) < 0.01)//kalo lebih besar dari batas toleransi, jaraknya naik
					distance++;
			}else{ //sama atau kagak
				if (first.stringValue(j).equals(second.stringValue(j)))
					distance++;
			}
		}
		return distance;
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
