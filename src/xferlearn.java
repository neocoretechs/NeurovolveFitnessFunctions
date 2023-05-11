
import static cnn.tools.Util.checkNotEmpty;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

import com.neocoretechs.neurovolve.NeuralNet;
import com.neocoretechs.neurovolve.Neurosome;
import com.neocoretechs.neurovolve.NeurosomeInterface;
import com.neocoretechs.neurovolve.activation.ActivationInterface;
import com.neocoretechs.neurovolve.activation.SoftMax;
import com.neocoretechs.neurovolve.fitnessfunctions.NeurosomeFitnessFunction;
import com.neocoretechs.neurovolve.fitnessfunctions.NeurosomeTransferFunction;
import com.neocoretechs.neurovolve.properties.LoadProperties;
import com.neocoretechs.neurovolve.properties.Props;
import com.neocoretechs.neurovolve.relatrix.Storage;
import com.neocoretechs.neurovolve.relatrix.ArgumentInstances;
import com.neocoretechs.neurovolve.worlds.World;
import com.neocoretechs.relatrix.DuplicateKeyException;
import com.neocoretechs.relatrix.client.RelatrixClient;
import com.neocoretechs.relatrix.client.RelatrixClientInterface;
import com.neocoretechs.relatrix.client.RemoteStream;
import com.neocoretechs.relatrix.client.RemoteTailSetIterator;

import cnn.components.Plate;
import cnn.driver.Dataset;
import cnn.driver.Instance;

import cnn.tools.Util;

/**
 * transfer learning using output layer from existing neurosome data as input for further evolution
 * @author Jonathan Groff (C) NeoCoreTechs 2023
 *
 */
public class xferlearn extends NeurosomeTransferFunction {
	private static final long serialVersionUID = -4154985360521212822L;
	private static boolean DEBUG = false;
	private static String prefix = "/media/jg/tensordisk/images/trainset/";//"D:/etc/images/trainset/";
	private static String localNode = "192.168.1.153";//"COREPLEX";
	private static String remoteNode = "192.168.1.153";//"COREPLEX";
	private String sguid;
	private static int dbPort = 9020;
    //private static Object mutex = new Object();
    private static float breakOnAccuracyPercentage = .9f; // set to 0 for 100% accuracy expected
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
	int ivec = 0;
	public static double[][] imageVecs; // each image output from previous neurosome, as 1D vector
	public static double[][] outputVecs; // each image output from previous neurosome, as 1D vector
	private static String[] imageLabels;
	private static String[] imageFiles;
	private static NeurosomeInterface solver = null;
	private static double improvementThreshold = .1; //percentage of improvement to return true in comparison of accuracy
	
	/**
	 * This ctor gets called by default by Genome, overloaded ctor with guid will get called by GenomeTransfer
	 * @param w
	 */
	public xferlearn(World w) {
		super(w);
	}
	/**
	 * This ctor gets called for transfer after default with world
	 * @param guid
	 */
	public xferlearn(World w, String sguid) {
		super(w);
		this.sguid = sguid;
		init();
	}

	public xferlearn() {}
	
	public void init() {
		
		if(datasetSize == 0) {
			Dataset dataset = Util.loadDataset(new File(prefix), null, false);
			datasetSize = dataset.getSize();
			// MinRawFitness is steps * testPerStep args one and two of setStepFactors
			getWorld().setStepFactors((float)datasetSize, 1.0f);
			System.out.printf("Dataset from %s loaded with %d images%n", prefix, datasetSize);
			createImageVecs(getWorld(), dataset);
			// set the properties to hardwire the source activation function
			LoadProperties.brandomizeActivation = false;
			// retrieve original solver
			solver = (Neurosome) Storage.loadSolver(getWorld().getRemoteStorageClient(), sguid);
			if(solver == null) {
				throw new RuntimeException("Could not locate "+sguid+" in stored solvers!");
			}
			// Now generate vectors of output of inference for stored solver to use as input to evolve
			// new solvers
		    for(int step = 0; step < getWorld().MaxSteps; step++) {
		    	double[] outVec = solver.execute(imageVecs[step]);
		    	outputVecs[step] = outVec;
		    }
		    // Make sure we set the activation function to be the same as original solver
			LoadProperties.sactivationFunction = ((NeuralNet)solver).getActivationFunction().getClass().getName();
			try {
				LoadProperties.activate = (ActivationInterface) Class.forName(LoadProperties.sactivationFunction).newInstance();
			} catch (SecurityException | ClassNotFoundException | InstantiationException | IllegalAccessException e) {
				throw new RuntimeException(e);
			}
			System.out.println("Activation function set to "+LoadProperties.sactivationFunction);
			// 
	
		}
	}
	    	
	@Override
	/**
	 * Compute cross entropy loss, return cost
	 */
	public Object execute(NeurosomeInterface ind) {
		//Long tim = System.currentTimeMillis();
		//System.out.println("Exec "+Thread.currentThread().getName()+" for ind "+ind.getName());
	 	float hits = 0;
        int errCount = 0;

        boolean[][] results = new boolean[(int)getWorld().MaxSteps][(int)getWorld().TestsPerStep];
        double cost = 0;
	    for(int test = 0; test < getWorld().TestsPerStep ; test++) {
	    	for(int step = 0; step < getWorld().MaxSteps; step++) {
	    		//System.out.println("Test:"+test+"Step:"+step+" "+ind);
	    		double[] outVec = ind.execute(outputVecs[step]);
	    		//for(int i = 0; i < outVec.length; i++)
	    			//if(Double.isNaN(outVec[i]))
	    				//outVec[i] = Double.MIN_VALUE;
	    		double[] actual = softMax(outVec);
	    		// expected is one-hot encoded for class
	    		double expected = 0;
	    		for(int j = 0; j < actual.length; j++) {
	    			expected = categoryNames.get(j).equals(imageLabels[step]) ? 1 : 0;
	    			cost += -(expected * Math.log(actual[j]) + (1 - expected) * Math.log(1 - actual[j]));
	    		}
	    		String predicted = classify(outVec);
	    		if(!predicted.equals(imageLabels[step])) {
	    			//if(predicted.equals("N/A"))
	    				//System.out.println("ENCOUNTERED N/A AT INDEX:"+step+" FOR:"+imageLabels[step]+" "+ind+" "+Thread.currentThread().getName()+" "+Arrays.toString(outVec));
	    			errCount++;
	    		} else {
	    			++hits;
	    			results[step][test] = true;
	    		}
	    	}
	    }
	    if(/*cost < 0 || Double.isInfinite(cost) ||*/ Double.isNaN(cost))
	    	cost = Double.MAX_VALUE/2;

	    //cost = ind.weightDecay(cost, .00001);

		if(World.SHOWTRUTH)
			System.out.println("ind:"+ind+" hits:"+hits+" err:"+errCount+" "+(hits/world.MinCost)*100+"%");
         // rawFit = world.MinRawFitness - hits;
         // break at predetermined accuracy level? adjust rawfit to 0 on that mark
         // MaxSteps * TestsPerStep is MinRawFitness. hits / MinRawFitness  = percentage passed
         if( breakOnAccuracyPercentage > 0 && (hits/(world.MaxSteps*world.TestsPerStep)) >= breakOnAccuracyPercentage) {
        	 getWorld().showTruth(ind, cost, results);
        	 System.out.println("Fitness function accuracy of "+breakOnAccuracyPercentage*100+"% equaled/surpassed by "+(hits/(getWorld().MaxSteps*getWorld().TestsPerStep))*100+"%.");
        	 //if(world.getRemoteStorageClient() != null) {
        		 //Storage.storeSolver(world.getRemoteStorageClient(), ind);
        	 //}
         } else {
        	 getWorld().showTruth(ind, cost, results);
         }
     	 //System.out.println("Exit "+Thread.currentThread().getName()+" for ind "+ind.getName()+" in "+(System.currentTimeMillis()-tim));
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
	    outputVecs = new double[(int)world.MaxSteps][];
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
    	}
	}
	
	
	/**
	 * Generates transfer learning multi task data.
	 * Reads guid Neurosome from db ri, generates output from dataset, writes each output vector to db ro
	 * @throws IOException 
	 * @throws IllegalAccessException 
	 * @throws ClassNotFoundException 
	 * @throws IllegalArgumentException 
	 */
	public static double testData(RelatrixClient ri, RelatrixClient ro, NeurosomeInterface ni, boolean verbose) throws IllegalArgumentException, ClassNotFoundException, IllegalAccessException, IOException {
		int errCount = 0;
		for(int j = 0; j < imageLabels.length; j++) {
			double[] outNeuro = ni.execute(imageVecs[j]);
			System.out.println(/*"Input "+img.toString()+*/" Output:"+Arrays.toString(outNeuro));
			Object[] o = new Object[outNeuro.length];
			for(int i = 0; i < outNeuro.length; i++) {
				o[i] = new Double(outNeuro[i]);
			}
			ArgumentInstances ai = new ArgumentInstances(o);
			try {
				ro.store(ni.getRepresentation(), imageLabels[j], ai);
				System.out.println(imageLabels[j]+" Stored!");
			} catch (IllegalAccessException | IOException | DuplicateKeyException e) {
				e.printStackTrace();
			}
			String predicted = classify(outNeuro);
			if (!predicted.equals(imageLabels[j])) {
				errCount++;
			}	
			if (verbose) {
				System.out.printf("Predicted: %s\t\tActual:%s\n", predicted, imageLabels[j]);
			}
		}
		
		double accuracy = ((double) (imageLabels.length - errCount)) / imageLabels.length;
		if (verbose) {
			System.out.printf("Final accuracy was %.9f\n", accuracy);
		}
		return accuracy;
	}
	/**
	 * Load imageVecs with previously stored output inference data from a specific 
	 * neurosome indexed by guid for the purpose of transfer learning.
	 * sguid field contains specific guid. dbPort contains port of remote db
	 * containing stored data. Expected format of db data is: [guid, image_file, double output array]
	 */
	private void loadStoredInference() {
		RemoteStream rs = null;
		try {
			RelatrixClient rkvc = new RelatrixClient(localNode, remoteNode, dbPort);
			rs = rkvc.findSetStream(sguid, "?", "?");
		} catch (IllegalArgumentException | ClassNotFoundException | IllegalAccessException | IOException e1) {
			throw new RuntimeException(e1);
		}
		datasetSize = (int) rs.of().count();
		imageVecs = new double[datasetSize][];
		imageLabels = new String[datasetSize];
		imageFiles = new String[datasetSize];
		rs.of().forEach(e -> {
			Comparable[] c = (Comparable[])e;
			Object[] o = ((ArgumentInstances)c[1]).argInst;
			imageVecs[ivec] = new double[o.length];
			for(int j = 0; j < o.length; j++)
				imageVecs[ivec][j] = ((Double)o[j]).doubleValue();
			int locationOfUnderscoreImage = ((String)c[0]).indexOf("_image");
			String name = ((String)c[0]);
			imageFiles[ivec] = ((String)c[0]);
			if(locationOfUnderscoreImage == -1)
				name = "UNNOWN";
			else
				name = name.substring(0, locationOfUnderscoreImage);
			imageLabels[ivec] = name;
			System.out.println(ivec+" = "+sguid+": "+imageLabels[ivec]+" | "+Arrays.toString(imageVecs[ivec]));
			++ivec;
		});
		
		System.out.printf("Dataset from %s loaded with %d images%n", sguid, datasetSize);
		// Construct a new world to spin up remote connection
		//categoryNames.get(index).getName() is category
		// MinRawFitness is steps * testPerStep args one and two of setStepFactors
		getWorld().setStepFactors((float)datasetSize, 1.0f);
	}
	/**
	 * Store the output from the passed Neurosome after it has processed data in imageVecs array.
	 * format of DB is [GUID, image file name, double output array from inference]
	 * @param ro
	 * @param ind
	 */
	private void storeInferredOutput(RelatrixClient ro, NeurosomeInterface ind) {
		for (int step = 0; step < imageVecs.length; step++) {
			double[] outNeuro = ind.execute(imageVecs[step]);
			System.out.println(/*"Input "+img.toString()+*/" Output:"+Arrays.toString(outNeuro));
			//System.out.println(/*"Input "+img.toString()+*/" Output:"+Arrays.toString(outNeuro));
			Object[] o = new Object[outNeuro.length];
			for(int i = 0; i < outNeuro.length; i++) {
				o[i] = new Double(outNeuro[i]);
			}
			ArgumentInstances ai = new ArgumentInstances(o);
			try {
				//String fLabel = String.format("%05d %s",step,imageLabels[step]);
				ro.store(ind.toString(), imageFiles[step], ai);
				//System.out.println(imageLabels[step]+" Stored!");
			} catch (IllegalAccessException | IOException | DuplicateKeyException e) {
				e.printStackTrace();
			}
		}
		System.out.println(this.getClass().getName()+" transfer data stored.");
	}
	
	@Override
	/**
	* Concatenate original solver and evolved solver from result of this evolutionary run.
	* Old solver is preserved from above loading in init()
	* Store it in Db to which client is connected
	* @param ro The client of remote db to store new solver
	* @param ind the new best individual from runs.
	* @return true if improvement of passed best solver concatenated with stored solver exceeds improvementThreshold value as percentage over stored solver
	*/
	public boolean transfer(RelatrixClientInterface ro, NeurosomeInterface ind) {
		NeurosomeInterface newSolver = solver.concat(ind);
		boolean isBetter = xferTests(newSolver);
		if(ro != null && isBetter) {
			Storage.storeSolver(ro, newSolver);
		}
		return isBetter;
	}
	
	/**
	 * Compares stored solver with passed presumed best individual against dataset input test vectors
	 * to determine if new best individual exceeds accuracy of stored solver by given percentage.
	 * @param nt The new concatenated Neurosome from Solver + best fit
	 * @return true if passed Neurosome is more accurate by given threshold. 
	 */
	public boolean xferTests(NeurosomeInterface nt)  {
		int nInErr = 0;
		int oInErr = 0;
		int total = 0;
		//NeuralNet.SHOWEIGHTS = true;
		//System.out.println("Neurosome original: "+(solver == null ? "NULL" : solver.getRepresentation()));
		//System.out.println("Neurosome concat: "+(nt == null? "NULL" : nt.getRepresentation()));
		//System.out.println("world "+(getWorld() == null ? "WORLD NULL" : getWorld()));
    	for(int step = 0; step < getWorld().MaxSteps; step++) {
			double[] outNeuro1 = solver.execute(imageVecs[step]);
			//System.out.println("Input "+imageLabels[step]+" Output1:"+Arrays.toString(outNeuro1));
			// chain the output
			//double[] outNeuro = nt.execute(outNeuro1);	
			String opredicted = classify(outNeuro1);
			if (!opredicted.equals(imageLabels[step])) {
				++oInErr;
			}	
			//System.out.printf("Predicted: %s\t\tActual:%s cat=%d File:%s\n", opredicted, imageLabels[step],categoryNames.indexOf(imageLabels[step]), imageFiles[step]);
			double[] outNeuro = nt.execute(imageVecs[step]);
			//System.out.println("Input "+imageLabels[step]+" Output2:"+Arrays.toString(outNeuro));		
			String predicted = classify(outNeuro);
			if (!predicted.equals(imageLabels[step])) {
				++nInErr;
			}	
			//System.out.printf("Predicted: %s\t\tActual:%s cat=%d File:%s\n", predicted, imageLabels[step],categoryNames.indexOf(imageLabels[step]), imageFiles[step]);
			++total;
		}	
		double oaccuracy = ((double) (total - oInErr)) / (double)total;
		double naccuracy = ((double) (total - nInErr)) / (double)total;
		double improvement = naccuracy - oaccuracy;
		System.out.println("Improvement of best = "+improvement);
		return (improvement >= improvementThreshold);
	}
	
}

