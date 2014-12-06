package clusterer;

import java.util.ArrayList;
import weka.core.Instance;
import weka.core.InstanceComparator;

class Cluster implements Cloneable {
	Cluster left;
	Cluster right;
	ArrayList<Instance> elements;
	ArrayList<Instance> lastelements;//simpen instances sebelum nya , kalo yg baru sama dengan yg lama algo selesai
	Instance centroid;
	double epsilon=0.01;//
	int height;

	Cluster() {
		elements = new ArrayList<Instance>();
		System.out.println("hello");
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
	void set_centroid(){
		centroid = new Instance(elements.get(0));
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
						centroid.setValue(j, count[j]/elements.size());
					//else //ini yang datanya musti count
				}
			}
		}
	}
	
	int getDistance(Instance i){
		int count = 0;
		for(int j=0;j<centroid.numAttributes();j++){
			if(centroid.attribute(j).isNumeric()){
				if(Math.abs(i.value(j) - centroid.value(j)) < epsilon)//kalo lebih besar dari batas toleransi, jaraknya naik
					count++;
			}else{ //sama atau kagak
				if (i.stringValue(j).equals(centroid.stringValue(j)))
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
				int count =0;
				InstanceComparator IC = new InstanceComparator();
				for(int i=0;i<elements.size();i++)
					count +=IC.compare(elements.get(i), elements.get(i));
				if (count == 0)
					convergen = true;
			}
		return convergen;
	}
	
}