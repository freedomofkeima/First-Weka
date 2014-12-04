package clusterer;

/**
 * Custom Partitional Clustering Implementation (implements Clusterer)
 * 
 * @author WbTeladan
 * @version 0.1, by WbTeladan @since December 4, 2014
 *
 */

public class myPartitionalClusterer extends RandomizableClusterer
implements NumberOfClustersRequestable, WeightedInstancesHandler, TechnicalInformationHandler {

//cara kerja k-means
//TENTUIN BAKAL ADA BERAPA CLUSTER
//Randomize Center nya yg mana aja -> attribut center nyolong dari data yg di randomize
//sisa data diclustering ke center2 yg terdekat -> buat algoritma comparision nya
//kalo udah semua di clustering, hitung ulang centernya -> pake fungsi ngitung center berdasarkan data yang ada -> sama kayak fungsi awal pas randomize
//hitung jarak isi cluster ke center sendiri dan center orang lain, kalo center orang lain lebih deket -> pindahin ke sana
//ulangi dari tahap 2
//tresshold kalo gak ada yg berubah2 clusternya atau udh x iterasi(maunya)

}
