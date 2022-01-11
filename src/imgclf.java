
import static cnn.tools.Util.checkNotEmpty;

import java.io.File;
import java.io.IOException;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import com.neocoretechs.neurovolve.Neurosome;
import com.neocoretechs.neurovolve.fitnessfunctions.NeurosomeFitnessFunction;
import com.neocoretechs.neurovolve.properties.LoadProperties;
import com.neocoretechs.neurovolve.worlds.RelatrixWorld;
import com.neocoretechs.neurovolve.worlds.World;

import com.neocoretechs.relatrix.client.RelatrixClient;

import cnn.components.Plate;
import cnn.driver.Dataset;
import cnn.driver.Instance;

import cnn.tools.Util;

/**
 * Fitness function to evolve image recognizer for imgclf datasets and perform same inference as
 * backpropagated solver control instance.
 * Hardwire port 9010 on same remote node as properties file to retrieve images and send to neurovolve
 * @author Jonathan Groff (C) NeoCoreTechs 2022
 *
 */
public class imgclf extends NeurosomeFitnessFunction {
	private static final long serialVersionUID = -4154985360521212822L;
	private static boolean DEBUG = false;
	private static String prefix = "D:/etc/images/trainset/";
    private static Object mutex = new Object();
    private World world;
	Dataset dataset;
	// We'll hardwire these in, but more robust code would not do so.
	private static enum Category {
		airplanes, butterfly, flower, grand_piano, starfish, watch
	};

	public static int NUM_CATEGORIES = Category.values().length;

	// Store the categories as strings.
	public static List<String> categoryNames = new ArrayList<>();
	static {
		for (Category cat : Category.values()) {
			categoryNames.add(cat.toString());
		}
	}
	
	/**
	 * @param guid
	 */
	public imgclf(World w, String guid) {
		super(w, guid);
		this.world = w;
		init();
	}
	/**
	 * @param argTypes
	 * @param returnType
	 */
	public imgclf(World w) {
		super(w);
		this.world = w;
		init();
	}

	public imgclf() {}
	
	public void init() {
		try {
			RelatrixWorld.ri = new RelatrixClient(LoadProperties.slocallIP, LoadProperties.sremoteIp, 9010);
		} catch (IOException e2) {
			throw new RuntimeException();
		}

		//if(args.length < 2)
		//	throw new Exception("Usage:java Infer <LocalIP Client> <Remote IpServer> <DB Port> <GUID of Neurosome> <Image file or directory>");
		//new RelatrixClient(args[0], args[1], Integer.parseInt(args[2]));
		dataset = Util.loadDataset(new File(prefix), null, false);
		System.out.printf("Dataset from %s loaded with %d images%n", prefix, dataset.getSize());
		// Construct a new world to spin up remote connection
		//categoryNames.get(index).getName() is category
		((RelatrixWorld)world).setStepFactors(dataset.getSize(), 1);
	}
	    	
	@Override
	public Object execute(Neurosome ind) {
			
	 	float hits = 0;
        float rawFit = -1;
        int errCount = 0;
        List<Instance> images = dataset.getImages();
        boolean[][] results = new boolean[(int) ((RelatrixWorld)world).MaxSteps][(int) ((RelatrixWorld)world).TestsPerStep];
        
	    for(int test = 0; test < ((RelatrixWorld)world).TestsPerStep ; test++) {
	    	for(int step = 0; step < ((RelatrixWorld)world).MaxSteps; step++) {
	    		//System.out.println("Test:"+test+"Step:"+step+" "+ind);
	    		Instance img = images.get(step);
	    		Plate[] plates = instanceToPlate(img);
	    		double[] d = packPlates(Arrays.asList(plates));
	    		float[] inFloat = new float[d.length];
	    		for(int i = 0; i < d.length; i++)
	    			inFloat[i] = (float) d[i];
	    		float[] outNeuro = ind.execute(inFloat);
	    		String predicted = classify(img, outNeuro);
	    		if (!predicted.equals(img.getLabel())) {
	    			errCount++;
	    		} else {
	    			++hits;
	    			results[step][test] = true;
	    		}
	    	}
	      }
		if(World.SHOWTRUTH)
			System.out.println("ind:"+ind+" hits:"+hits+" err:"+errCount+" "+(hits/dataset.getSize())*100+"%"/*"Input "+img.toString()+*/);
         //if( al.data.size() == 1 && ((Strings)(al.data.get(0))).data.equals("d")) hits = 10; // test
         rawFit = (((RelatrixWorld)world).MinRawFitness - hits);
         // The SHOWTRUTH flag is set on best individual during run. We make sure to 
         // place the checkAndStore inside the SHOWTRUTH block to ensure we only attempt to process
         // the best individual, and this is what occurs in the showTruth method
         if(.8 <= (hits/dataset.getSize()))
        	 rawFit = 0;
         ((RelatrixWorld)world).showTruth(ind, rawFit, results);
         // break at 80% success
         return rawFit;
	}

	
	/** Returns the predicted label for the image. */
	public static String classify(Instance img, float[] probs) {
		double maxProb = -1;
		int bestIndex = -1;
		for (int i = 0; i < probs.length; i++) {
			if (probs[i] > maxProb) {
				maxProb = probs[i];
				bestIndex = i;
			}
		}
		if(bestIndex == -1)
			return "N/A";
		return categoryNames.get(bestIndex);
	}
	
	private static Plate[] instanceToPlate(Instance instance) {
			return new Plate[] {
					new Plate(intImgToDoubleImg(instance.getRedChannel())),
					new Plate(intImgToDoubleImg(instance.getBlueChannel())),
					new Plate(intImgToDoubleImg(instance.getGreenChannel())),
					new Plate(intImgToDoubleImg(instance.getGrayImage())),
			};
	}
	
	private static double[][] intImgToDoubleImg(int[][] intImg) {
		double[][] dblImg = new double[intImg.length][intImg[0].length];
		for (int i = 0; i < dblImg.length; i++) {
			for (int j = 0; j < dblImg[i].length; j++) {
				dblImg[i][j] = ((double) 255 - intImg[i][j]) / 255;
			}
		}
		return dblImg;
	}
	
	/** 
	 * Pack the plates into a single, 1D double array. Used to connect the plate layers
	 * with the fully connected layers.
	 */
	private static double[] packPlates(List<Plate> plates) {
		checkNotEmpty(plates, "Plates to pack", false);
		int flattenedPlateSize = plates.get(0).getTotalNumValues();
		double[] result = new double[flattenedPlateSize * plates.size()];
		for (int i = 0; i < plates.size(); i++) {
			System.arraycopy(
					plates.get(i).as1DArray(),
					0 /* Copy the whole flattened plate! */,
					result,
					i * flattenedPlateSize,
					flattenedPlateSize);
		}
		return result;
	}
	
}

