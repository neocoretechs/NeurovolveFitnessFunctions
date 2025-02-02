
import static cnn.tools.Util.checkNotEmpty;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import com.neocoretechs.neurovolve.NeurosomeInterface;
import com.neocoretechs.neurovolve.activation.SoftMax;
import com.neocoretechs.neurovolve.fitnessfunctions.NeurosomeFitnessFunction;
import com.neocoretechs.neurovolve.fitnessfunctions.NeurosomeTransferFunction;
import com.neocoretechs.neurovolve.relatrix.ArgumentInstances;
import com.neocoretechs.neurovolve.relatrix.Storage;
import com.neocoretechs.neurovolve.worlds.World;
import com.neocoretechs.relatrix.DuplicateKeyException;
import com.neocoretechs.relatrix.client.RelatrixClient;
import com.neocoretechs.relatrix.client.RelatrixKVClientInterface;

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
public class imgclf extends NeurosomeTransferFunction {
	private static final long serialVersionUID = -4154985360521212822L;
	private static boolean DEBUG = false;
	private static String prefix = "C:/etc/images/trainset/";//"/media/jg/tensordisk/images/trainset/";//
    //private static Object mutex = new Object();
    private static float breakOnAccuracyPercentage = .7f; // set to 0 for 100% accuracy expected
	//Dataset dataset;
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
	
	public static int datasetSize = 0;
	public static double[][] imageVecs; // each image as 1D float vector
	private static String[] imageLabels;
	private static String[] imageFiles;
	
	/**
	 * @param guid
	 */
	public imgclf(World w) {
		super(w);
		init();
	}

	public imgclf() {}
	
	public void init() {

		//if(args.length < 2)
		//	throw new Exception("Usage:java Infer <LocalIP Client> <Remote IpServer> <DB Port> <GUID of Neurosome> <Image file or directory>");
		//new RelatrixClient(args[0], args[1], Integer.parseInt(args[2]));
		if(datasetSize == 0) {
			Dataset dataset = Util.loadDataset(new File(prefix), null, false);
			datasetSize = dataset.getSize();
			System.out.printf("Dataset from %s loaded with %d images%n", prefix, datasetSize);
			// Construct a new world to spin up remote connection
			//categoryNames.get(index).getName() is category
			// MinRawFitness is steps * testPerStep args one and two of setStepFactors
			getWorld().setStepFactors((float)datasetSize, 1.0f);
			createImageVecs(getWorld(), dataset);
		}
	}
	    	
	@Override
	/**
	 * Compute cross entropy loss, return cost
	 */
	public Object execute(NeurosomeInterface ind) {
		Long tim = System.currentTimeMillis();
		System.out.println("Exec "+Thread.currentThread().getName()+" for ind "+ind.getName());
	 	//float hits = 0;
        //int errCount = 0;

        //boolean[][] results = new boolean[(int)getWorld().MaxSteps][(int)getWorld().TestsPerStep];
        double cost = 0;
	    for(int test = 0; test < getWorld().TestsPerStep ; test++) {
	    	for(int step = 0; step < getWorld().MaxSteps; step++) {
	    		//System.out.println("Test:"+test+"Step:"+step+" "+ind);
	    		double[] outVec = ind.execute(imageVecs[step]);
	    		double[] actual = softMax(outVec);
	    		// expected is one-hot encoded for class
	    		double expected = 0;
	    		for(int j = 0; j < actual.length; j++) {
	    			expected = categoryNames.get(j).equals(imageLabels[step]) ? 1 : 0;
	    			cost += -(expected * Math.log(actual[j]) + (1 - expected) * Math.log(1 - actual[j]));
	    		}
	    		//String predicted = classify(outVec, actual);
	    		//if(!predicted.equals(imageLabels[step])) {
	    			//if(predicted.equals("N/A"))
	    				//System.out.println("ENCOUNTERED N/A AT INDEX:"+step+" FOR:"+imageLabels[step]+" "+ind+" "+Thread.currentThread().getName()+" "+Arrays.toString(outVec));
	    			//errCount++;
	    		//} else {
	    			//++hits;
	    			//results[step][test] = true;
	    		//}
	    	}
	    }
	    if(/*cost < 0 || Double.isInfinite(cost) ||*/ Double.isNaN(cost))
	    	cost = Double.MAX_VALUE/2;

	    //cost = ind.weightDecay(cost, .00001);

	    /*
		if(World.SHOWTRUTH)
			System.out.println("ind:"+ind+" hits:"+hits+" err:"+errCount+" "+(hits/ind.getWorld().MinCost)*100+"%");
         if( breakOnAccuracyPercentage > 0 && (hits/(getWorld().MaxSteps*getWorld().TestsPerStep)) >= breakOnAccuracyPercentage) {
        	 getWorld().showTruth(ind, cost, results);
        	 System.out.println("Fitness function accuracy of "+breakOnAccuracyPercentage*100+"% equaled/surpassed by "+(hits/(getWorld().MaxSteps*getWorld().TestsPerStep))*100+"%.");
         } else {
        	 getWorld().showTruth(ind, cost, results);
         }
         */
     	 System.out.println("Exit "+Thread.currentThread().getName()+" for ind "+ind.getName()+" in "+(System.currentTimeMillis()-tim));
         return cost;
	}

	
	/** Returns the predicted label for the image. */
	public static String classify(double[] dprobs) {
		double maxProb = -1;
		int bestIndex = -1;
		SoftMax sf = new SoftMax(dprobs);
		for (int i = 0; i < dprobs.length; i++) {
			double smax = sf.activate(dprobs[i]);
			// normalize
			//if(smax < 0 || Double.isNaN(smax))
				//smax = Double.MIN_VALUE;
			if (smax > maxProb) {
				maxProb = smax;
				bestIndex = i;
			}
		}
		if(bestIndex == -1)
			return "N/A";
		return categoryNames.get(bestIndex);
	}
	
	/** Returns the predicted label for the image. */
	public static String classify(double[] dprobs, double[] sf) {
		double maxProb = -1;
		int bestIndex = -1;
		for (int i = 0; i < dprobs.length; i++) {
			if (sf[i] > maxProb) {
				maxProb = sf[i];
				bestIndex = i;
			}
		}
		if(bestIndex == -1)
			return "N/A";
		return categoryNames.get(bestIndex);
	}
	
	public static double[] softMax(double[] dprobs) {
		SoftMax sf = new SoftMax(dprobs);
		double[] smax = new double[dprobs.length];
		for (int i = 0; i < dprobs.length; i++) {
			smax[i] = sf.activate(dprobs[i]);
			// normalize
			//if(smax[i] < 0 || Double.isNaN(smax[i]))
				//smax[i] = Double.MIN_VALUE;
		}
		return smax;
	}
	
	private static Plate[] instanceToPlate(Instance instance) {
			return new Plate[] {
					//new Plate(intImgToDoubleImg(instance.getRedChannel())),
					//new Plate(intImgToDoubleImg(instance.getBlueChannel())),
					//new Plate(intImgToDoubleImg(instance.getGreenChannel())),
					new Plate(intImgToDoubleImg(instance.getGrayImage())),
			};
	}
	/**
	 * Scale the integer image channel which is byte 0-255 to double 0 to 1 range
	 * @param intImg
	 * @return
	 */
	private static double[][] intImgToDoubleImg(int[][] intImg) {
		double[][] dblImg = new double[intImg.length][intImg[0].length];
		for (int i = 0; i < dblImg.length; i++) {
			for (int j = 0; j < dblImg[i].length; j++) {
				dblImg[i][j] = ((double)(255 - intImg[i][j])) / ((double)255);
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
	
	private static void createImageVecs(World world, Dataset dataset) {
	    imageVecs = new double[(int)world.MaxSteps][];
	    imageLabels = new String[(int)world.MaxSteps];
	    imageFiles = new String[(int)world.MaxSteps];
	    List<Instance> images = dataset.getImages();
    	for(int step = 0; step < world.MaxSteps; step++) {
    		//System.out.println("Test:"+test+"Step:"+step+" "+ind);
    		Instance img = images.get(step);
    		Plate[] plates = instanceToPlate(img);
       		imageLabels[step] = img.getLabel();
       		imageFiles[step] = img.getName();
    		imageVecs[step] = packPlates(Arrays.asList(plates));
    		/*
    		float[] inFloat = new float[img.getWidth()*img.getHeight()];
    		int[] dstBuff = new int[img.getWidth()*img.getHeight()];
    		Instance.readLuminance(img.getImage(), dstBuff);
    		int i = 0;
    		for (int row = 0; row < img.getHeight(); ++row) {
    			for (int col = 0; col < img.getWidth(); ++col) {
    				inFloat[i] = ((float)dstBuff[i]) / 255.0f;
	    			//System.out.println(i+"="+inFloat[i]);
	    			++i;
    			}
    		}
    		float[] inFloat = new float[d.length];
    		*/
    	}
	}
	
	@Override
	/**
	 * Generates transfer learning multi task data. Alternate to generation from image directory.
	 * Generates output from each inference of passed neurosome against imageVecs training data.
	 * @param ind Neurosome to perform inference with each imageVecs vector
	 * @return true since this function can also be used to continue until we reach a threshold, but here just return true to stop.
	 */
	public boolean transfer(NeurosomeInterface ind, NeurosomeInterface indo) {
		for (int step = 0; step < imageVecs.length; step++) {
			double[] outNeuro = ind.execute(imageVecs[step]);
			//System.out.println(/*"Input "+img.toString()+*/" Output:"+Arrays.toString(outNeuro));
			Object[] o = new Object[outNeuro.length];
			for(int i = 0; i < outNeuro.length; i++) {
				o[i] = new Double(outNeuro[i]);
			}
			ArgumentInstances ai = new ArgumentInstances(o);
			//try {
				//String fLabel = String.format("%05d %s",step,imageLabels[step]);
				//((RelatrixClient)ro).store(ind.toString(), imageFiles[step], ai);
				//System.out.println(imageLabels[step]+" Stored!");
			//} catch (IllegalAccessException | IOException | DuplicateKeyException e) {
			//	e.printStackTrace();
			//}
		}
		System.out.println(this.getClass().getName()+" transfer data stored.");
		return true;
	}
	
}

