
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

public class Dataset implements Serializable {
	private static final long serialVersionUID = 1L;
	// the list of all instances
	private ArrayList<RawInstance> instances;

	public Dataset() {
		this.instances = new ArrayList<>();
	}

	// get the size of the dataset
	public int getSize() {
		return instances.size();
	}

	// add instance into the data set
	public void add(RawInstance inst) {
		instances.add(inst);
	}

	// Return the list of images.
	public List<RawInstance> getImages() {
		return instances;
	}
	
	public static Dataset loadDataset(File dir, String label, boolean directoryIsLabel) {
		Dataset d = new Dataset();
		ArrayList<File> fileList = new ArrayList<File>();
		if(dir.isFile()) {
			fileList.add(dir);
		} else {
			for (File file : dir.listFiles()) {
				// check all files
				if (!file.isFile() || !file.getName().endsWith(".jpg")) {
					continue;
				}
				fileList.add(file);
			}
		}
		for(File fi: fileList ) {
			// String path = file.getAbsolutePath();
			try {
				System.out.println("Reading "+fi.getName());
				// load in all images
				
				// every image's name is in such format: label_image_XXXX(4 digits) though this code could handle more than
				// 4 digits.
				String name = fi.getName();
				int locationOfUnderscoreImage = name.indexOf("_image");

				if(label == null && directoryIsLabel)
					name = dir.getName();
				else
					if(label == null && locationOfUnderscoreImage == -1)
						name = "UNNOWN";
					else
						if(label == null)
							name = name.substring(0, locationOfUnderscoreImage);
						else
							name = label;
				
				RawInstance instance = new RawInstance(fi.getName(), 128, 128, 10000, name);
				FileInputStream fis = new FileInputStream(fi);
				instance.readImage(fis);
				fis.close();
				d.add(instance);

			} catch (IOException e) {
				System.err.println("Error: cannot load in the image file");
				System.exit(1);
			}
		}
		return d;
	}
}
