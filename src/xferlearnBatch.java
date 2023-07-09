
import static cnn.tools.Util.checkNotEmpty;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Stream;

import com.neocoretechs.neurovolve.GPListenerInterface;
import com.neocoretechs.neurovolve.NeurosomeInterface;
import com.neocoretechs.neurovolve.activation.ActivationInterface;
import com.neocoretechs.neurovolve.activation.SoftMax;
import com.neocoretechs.neurovolve.fitnessfunctions.NeurosomeTransferFunction;
import com.neocoretechs.neurovolve.multiprocessing.SynchronizedFixedThreadPoolManager;
import com.neocoretechs.neurovolve.properties.LoadProperties;
import com.neocoretechs.neurovolve.relatrix.Storage;
import com.neocoretechs.neurovolve.relatrix.ArgumentInstances;
import com.neocoretechs.neurovolve.worlds.World;

import com.neocoretechs.relatrix.DuplicateKeyException;
import com.neocoretechs.relatrix.client.RelatrixClient;

import cnn.components.Plate;
import cnn.driver.Dataset;
import cnn.driver.Instance;

import cnn.tools.Util;

/**
 * transfer learning using output layer from existing neurosome data as input for further evolution
 * @author Jonathan Groff (C) NeoCoreTechs 2023
 *
 */
public class xferlearnBatch extends NeurosomeTransferFunction {
	private static final long serialVersionUID = -4154985360521212822L;
	public static boolean DEBUG = false;
	public static boolean TIMING = false;
	private static String prefix = "C:/etc/images/trainset/";//"/media/jg/tensordisk/images/trainset/";
	//private static String localNode = "192.168.1.153";//"COREPLEX";
	//private static String remoteNode = "192.168.1.153";//"COREPLEX";
	private String sguid;
	//private static int dbPort = 9020;
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
	public static ArrayList<double[][]> imageVecs; // each image output from previous neurosome, as 1D vector
	//public static double[][] outputVecs; // each image output from previous neurosome, as 1D vector
	public static List<nOutput> outputNeuros;
	private static ArrayList<String[]> imageLabels;
	private static ArrayList<String[]> imageFiles;
    private static double[][] imageVecsArray = null;
    private static String[] imageLabelsArray = null;
    private static String[] imageFilesArray = null;
	private static AtomicInteger threadIndex = new AtomicInteger(0);
	//private static NeurosomeInterface solver = null;
	private static double improvementThreshold = LoadProperties.fOffspringImprovementFactor; //percentage of improvement to return true in comparison of accuracy
	
	static class nOutput {
		String guid;
		ActivationInterface[] activations; // each layer of source
		ArrayList<double[][]> outputVecs;
	}
	
	/**
	 * This ctor gets called by default by Genome, overloaded ctor with guid will get called by GenomeTransfer
	 * @param w
	 */
	public xferlearnBatch(World w) {
		super(w);
	}
	/**
	 * This ctor gets called for transfer after default with world
	 * @param guid
	 */
	public xferlearnBatch(World w, String sguid) {
		super(w);
		this.sguid = sguid;
		init();
	}

	public xferlearnBatch() {}
	
	public void init() {
		
		if(datasetSize == 0) {
			Dataset dataset = Util.loadDataset(new File(prefix), null, false);
			datasetSize = dataset.getSize();
			System.out.printf("Dataset from %s loaded with %d images%n", prefix, datasetSize);
			createImageVecs(getWorld(), dataset);
			// set the properties to hardwire the source activation function
			LoadProperties.brandomizeActivation = false;
			// retrieve original solver
			//solver = (Neurosome) Storage.loadSolver(getWorld().getRemoteStorageClient(), sguid);
			// if guid is 'all' load all
			long stim = System.currentTimeMillis();
			System.out.println("Loading solver(s)...");
			ArrayList<NeurosomeInterface> solvers = Storage.loadSolvers(sguid, LoadProperties.slearnDb);
			if(solvers.isEmpty()) {
				throw new RuntimeException("Could not locate "+sguid+" in stored solvers!");
			}
			System.out.println(solvers.size()+" Solver(s) loaded in "+(System.currentTimeMillis()-stim)+" ms.");
			// MinRawFitness is steps * testPerStep args one and two of setStepFactors
			//getWorld().setStepFactors((float)datasetSize, (float)solvers.size());
			outputNeuros = Collections.synchronizedList(new ArrayList<nOutput>());
			System.out.println("Building vector(s)...");
			// We can only use 1 CUDA thread initially due to high initial memory requirements using original solvers
			// Once transfer starts, we can increase thread level
			int permThreads = LoadProperties.iCUDAThreads;
			LoadProperties.iCUDAThreads = 2; // main thread plus compute thread
			stim = System.currentTimeMillis();
			/* main thread only exec
			for(NeurosomeInterface solver: solvers) {
				solver.init();
				// Now generate vectors of output of inference for stored solver to use as input to evolve
				// new solvers
				nOutput noutput = new nOutput();
				noutput.guid = solver.getName();
				noutput.activations = new ActivationInterface[solver.getLayers()];
				for(int i = 0; i < solver.getLayers(); i++) {
					noutput.activations[i] = solver.getActivationFunction(i);
				}
				// We have the image tests chunked for maximum batch size for each test.
				// Perform the batched chunk, get the results, then chunk them the same way in the
				// output vectors that will serve as inputs to the next round of batched chunks
				// in the main test fitness function body below.
				noutput.outputVecs = new ArrayList<double[][]>(imageVecs.size());
				long ttim = System.currentTimeMillis();
			    for(int test = 0; test < imageVecs.size() ; test++) {
			    	ArrayList<double[]> resVecs = solver.execute(imageVecs.get(test));
					double[][] resArray = new double[resVecs.size()][];
					for(int k = 0; k < resVecs.size(); k++) {
						resArray[k] = resVecs.get(k);
					}
			    	noutput.outputVecs.add(resArray);
			    }
				System.out.println(solver+" executed batches in "+(System.currentTimeMillis()-ttim)+" ms.");
				//noutput.outputVecs = new double[datasetSize][];
				//for(int step = 0; step < datasetSize; step++) {
				//	double[] outVec = solver.execute(imageVecs[step]);
				//	noutput.outputVecs[step] = outVec;
				//}
				outputNeuros.add(noutput);
			}
			*/
			//
			// Generate the full set of tests for each solver. The tests are chunked based on maximum batch size
			// and this is all set up in the createImageVecs method previously executed above.
			// The noutput.outputVecs list is chunked in the same manner are the test images, but these outputVecs
			// are the result of executing the batched tests against the solver contained in the noutput structure.
			//		
			Future<?>[] jobs = new Future[solvers.size()];
			threadIndex.set(0);
			for(int i = 0; i < solvers.size(); i++) {
			    	jobs[i] = SynchronizedFixedThreadPoolManager.submit(new Runnable() {
			    		@Override
			    		public void run() {
			    		    NeurosomeInterface solver = solvers.get(threadIndex.getAndIncrement());
			    			solver.init();
							// Now generate vectors of output of inference for stored solver to use as input to evolve
							// new solvers
							nOutput noutput = new nOutput();
							noutput.guid = solver.getName();
							noutput.activations = new ActivationInterface[solver.getLayers()];
							for(int i = 0; i < solver.getLayers(); i++) {
								noutput.activations[i] = solver.getActivationFunction(i);
							}
							noutput.outputVecs = new ArrayList<double[][]>(imageVecs.size());
							// We have the image tests chunked for maximum batch size for each test.
							// Perform the batched chunk, get the results, then chunk them the same way in the
							// output vectors that will serve as inputs to the next round of batched chunks
							// in the main test fitness function body below.
							long ttim = System.currentTimeMillis();
						    for(int test = 0; test < imageVecs.size() ; test++) {
						    	ArrayList<double[]> resVecs = solver.execute(imageVecs.get(test));
								double[][] resArray = new double[resVecs.size()][];
								for(int k = 0; k < resVecs.size(); k++) {
									resArray[k] = resVecs.get(k);
								}
						    	noutput.outputVecs.add(resArray);
						    }
						    if(TIMING)
						    	System.out.println(Thread.currentThread().getName()+" "+solver+" executed batches in "+(System.currentTimeMillis()-ttim)+" ms.");
							outputNeuros.add(noutput);
			    		} // run
			    	},"COMPUTE"); // spin
			} 
			SynchronizedFixedThreadPoolManager.waitForCompletion(jobs);
			
			// restore CUDA threads for transfer learning data level
			LoadProperties.iCUDAThreads = permThreads;
			System.out.println(outputNeuros.size()+" Vector(s) built in in "+(System.currentTimeMillis()-stim)+" ms.");
		}
	}
	    	
	@Override
	/**
	 * Compute cross entropy loss, return cost
	 */
	public Object execute(NeurosomeInterface ind) {
		Long tim = System.currentTimeMillis();
		if(TIMING)
			System.out.println("Exec "+Thread.currentThread().getName()+" id:"+Thread.currentThread().getId()+" for ind "+ind.getName());
	 	//float hits = 0;
        //int errCount = 0;
        double[] nCost = new double[outputNeuros.size()];
        //boolean[][] results = new boolean[(int)getWorld().MaxSteps][(int)getWorld().TestsPerStep];
        // outputNeuros is each parent Neurosome, and each one contains all tests batched to maximum GPU memory capacity
	    for(int test = 0; test < outputNeuros.size() ; test++) {
    		// all source neuros for this step against this target neuro
    		// find lowest cost
	    	nOutput noutput = outputNeuros.get(test);
			// set the current individual Neurosome activation function to original parent solver Neurosome activation function
			for(int i = 0; i < ind.getLayers(); i++)
				ind.setActivationFunction(i,noutput.activations[i]);
			// Perform each chunked test based on maximum batch size output vectors of parent neurosome we set up in init.
			// noutput.outputVecs contains results of inference of parent Neurosome on images,
			// batched as images were originally set up in createImageVecs.
	    	for(int step = 0; step < noutput.outputVecs.size(); step++) {
	    		//System.out.println("Test:"+test+"Step:"+step+" "+ind);
	    		// execute our current individual child candidate with our source candidate output vector
	    		ArrayList<double[]> outVecs = ind.execute(noutput.outputVecs.get(step));
	    		// iterate through each batch result
	        	for(int step2 = 0; step2 < outVecs.size(); step2++) {
		    		double[] actual = softMax(outVecs.get(step2));
		    		// expected is one-hot encoded for class
		    		double expected = 0;
		    		for(int j = 0; j < actual.length; j++) {
		    			expected = categoryNames.get(j).equals(imageLabels.get(step)[step2]) ? 1 : 0;
		    			nCost[test] += -(expected * Math.log(actual[j]) + (1 - expected) * Math.log(1 - actual[j]));
		    		}
	        	}
	    		//double[] actual = softMax(outVec);
	    		// expected is one-hot encoded for class
	    		//double expected = 0;
	    		//for(int j = 0; j < actual.length; j++) {
	    			//expected = categoryNames.get(j).equals(imageLabels[step]) ? 1 : 0;
	    			//nCost[test] += -(expected * Math.log(actual[j]) + (1 - expected) * Math.log(1 - actual[j]));
	    		//}
	    		//String predicted = classify(outVec, actual);
	    		//if(!predicted.equals(imageLabels[step])) {
	    			//if(predicted.equals("N/A"))
	    			//System.out.println("ENCOUNTERED N/A AT INDEX:"+step+" FOR:"+imageLabels[step]+" "+ind+" "+Thread.currentThread().getName()+" "+Arrays.toString(outVec));
	    			//errCount++;
	    		//} else {
	    		//	++hits;
	    			//results[step][test] = true;
	    		//}
	    	}
	    }
	    //
        double cost = Double.MAX_VALUE;
	    for(int test = 0; test < outputNeuros.size() ; test++) {
    		// all source neuros for this step against this target neuro
    		// find lowest cost
	    	nOutput noutput = outputNeuros.get(test);
		    if(Double.isNaN(nCost[test]))
		    	nCost[test] = Double.MAX_VALUE/2;
		    if(DEBUG)
		    	System.out.println("cost "+test+"="+nCost[test]+" ind="+ind+" source="+noutput.guid);
	    	if(nCost[test] < cost) {
	    		cost = nCost[test];
	    		// set our best candidate source guid to lowest cost so far
	    		setSourceGuid(noutput.guid);
			    if(DEBUG)
			    	System.out.println("set lowest cost "+test+"="+nCost[test]+" ind="+ind+" source="+noutput.guid);
	    	}
	    }

	    //cost = ind.weightDecay(cost, .00001);

		//if(World.SHOWTRUTH)
			//System.out.println("ind:"+ind+" hits:"+hits+" err:"+errCount+" "+(hits/world.MinCost)*100+"%");
         //if( breakOnAccuracyPercentage > 0 && (hits/(world.MaxSteps*world.TestsPerStep)) >= breakOnAccuracyPercentage) {
        	 //getWorld().showTruth(ind, cost, results);
        	 //System.out.println("Fitness function accuracy of "+breakOnAccuracyPercentage*100+"% equaled/surpassed by "+(hits/(getWorld().MaxSteps*getWorld().TestsPerStep))*100+"%.");
        	 //if(world.getRemoteStorageClient() != null) {
        		 //Storage.storeSolver(world.getRemoteStorageClient(), ind);
        	 //}
         //} else {
        	 //getWorld().showTruth(ind, cost, results);
         //}
	    if(TIMING)
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
	    imageVecs = new ArrayList<double[][]>((int)world.TestsPerStep);
	    imageLabels = new ArrayList<String[]>((int)world.TestsPerStep);
	    imageFiles = new ArrayList<String[]>((int)world.TestsPerStep);

	    List<Instance> images = dataset.getImages();
	    int maxImages = 0;
	    if(world.TestsPerStep > images.size())
	    	world.TestsPerStep = images.size();
	    System.out.println("Tests per step = "+ world.TestsPerStep);
	    // arrange entire test set into blocks in TestsPerStep which represents max number of tests per batch
	    while(maxImages < images.size()) {
	  		if((images.size() - maxImages) < world.TestsPerStep) {
	  			imageVecsArray = new double[(images.size() - maxImages)][];
	  			imageLabelsArray = new String[(images.size() - maxImages)];
	  			imageFilesArray = new String[(images.size() - maxImages)];
	  		} else {
	  			imageVecsArray = new double[(int) world.TestsPerStep][];
	  			imageLabelsArray = new String[(int) world.TestsPerStep];
	  			imageFilesArray = new String[(int) world.TestsPerStep];	
	  		}
	  		for(int step = 0; step < imageFilesArray.length; step++) {
	  			//System.out.println("Test:"+test+"Step:"+step+" "+ind);
	  			Instance img = images.get(maxImages++);
	  			Plate[] plates = instanceToPlate(img);
	  			imageLabelsArray[step] = img.getLabel();
	  			imageFilesArray[step] = img.getName();
	  			imageVecsArray[step] = packPlates(Arrays.asList(plates));
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
	  		imageVecs.add(imageVecsArray);
	  		imageLabels.add(imageLabelsArray);
	  		imageFiles.add(imageFilesArray);
	  		System.out.println("Image Vecs = "+imageVecs.size()+" Array Size="+imageVecs.get(imageVecs.size()-1).length);
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
	    double[][] imageVecsArray = null;
	    String[] imageLabelsArray = null;
	    String[] imageFilesArray = null;
		for(int i = 0; i < imageLabels.size(); i++) {
  			imageLabelsArray = imageLabels.get(i);
  			imageVecsArray = imageVecs.get(i);
		  for(int j = 0; j < imageLabelsArray.length; j++) {
			double[] outNeuro = ni.execute(imageVecsArray[j]);
			System.out.println(/*"Input "+img.toString()+*/" Output:"+Arrays.toString(outNeuro));
			Object[] o = new Object[outNeuro.length];
			for(int k = 0; k < outNeuro.length; k++) {
				o[k] = new Double(outNeuro[k]);
			}
			ArgumentInstances ai = new ArgumentInstances(o);
			try {
				ro.store(ni.getRepresentation(), imageLabelsArray[j], ai);
				System.out.println(imageLabelsArray[j]+" Stored!");
			} catch (IllegalAccessException | IOException | DuplicateKeyException e) {
				e.printStackTrace();
			}
			String predicted = classify(outNeuro);
			if (!predicted.equals(imageLabelsArray[j])) {
				errCount++;
			}	
			if (verbose) {
				System.out.printf("Predicted: %s\t\tActual:%s\n", predicted, imageLabelsArray[j]);
			}
		  }
		}
		double accuracy = ((double) (imageLabelsArray.length - errCount)) / imageLabelsArray.length;
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
		Stream rs = null;
		//try {
		//	RelatrixClient rkvc = new RelatrixClient(localNode, remoteNode, dbPort);
		//	rs = rkvc.findSetStream(sguid, "?", "?");
		//} catch (IllegalArgumentException | ClassNotFoundException | IllegalAccessException | IOException e1) {
		//	throw new RuntimeException(e1);
		//}	    double[][] imageVecsArray = null;

		for(int i = 0; i < imageLabels.size(); i++) {
  			imageLabelsArray = imageLabels.get(i);
  			imageVecsArray = imageVecs.get(i);
		//datasetSize = (int) Stream.of().count();
		imageVecs = new ArrayList<double[][]>();
		imageLabels = new ArrayList<String[]>();
		imageFiles = new ArrayList<String[]>();
		Stream.of().forEach(e -> {
			Comparable[] c = (Comparable[])e;
			Object[] o = ((ArgumentInstances)c[1]).argInst;
			imageVecsArray[ivec] = new double[o.length];
			for(int j = 0; j < o.length; j++)
				imageVecsArray[ivec][j] = ((Double)o[j]).doubleValue();
			int locationOfUnderscoreImage = ((String)c[0]).indexOf("_image");
			String name = ((String)c[0]);
			imageFilesArray[ivec] = ((String)c[0]);
			if(locationOfUnderscoreImage == -1)
				name = "UNNOWN";
			else
				name = name.substring(0, locationOfUnderscoreImage);
			imageLabelsArray[ivec] = name;
			System.out.println(ivec+" = "+sguid+": "+imageLabelsArray[ivec]+" | "+Arrays.toString(imageVecsArray[ivec]));
			++ivec;
		});
		}
		System.out.printf("Dataset from %s loaded with %d images%n", sguid, datasetSize);
		// Construct a new world to spin up remote connection
		//categoryNames.get(index).getName() is category
		// MinRawFitness is steps * testPerStep args one and two of setStepFactors
		//getWorld().setStepFactors((float)datasetSize, 1.0f);
	}
	/**
	 * Store the output from the passed Neurosome after it has processed data in imageVecs array.
	 * format of DB is [GUID, image file name, double output array from inference]
	 * @param ro
	 * @param ind
	 */
	private void storeInferredOutput(RelatrixClient ro, NeurosomeInterface ind) {
		for (int step = 0; step < imageVecsArray.length; step++) {
			double[] outNeuro = ind.execute(imageVecsArray[step]);
			System.out.println(/*"Input "+img.toString()+*/" Output:"+Arrays.toString(outNeuro));
			//System.out.println(/*"Input "+img.toString()+*/" Output:"+Arrays.toString(outNeuro));
			Object[] o = new Object[outNeuro.length];
			for(int i = 0; i < outNeuro.length; i++) {
				o[i] = new Double(outNeuro[i]);
			}
			ArgumentInstances ai = new ArgumentInstances(o);
			try {
				//String fLabel = String.format("%05d %s",step,imageLabels[step]);
				ro.store(ind.toString(), imageFilesArray[step], ai);
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
	* @param ind the new best individual from runs.
	* @param solver Original parent seed solver loaded from db
	* @return true if improvement of passed best solver concatenated with stored solver exceeds improvementThreshold value as percentage over stored solver
	*/
	public boolean transfer(NeurosomeInterface ind, NeurosomeInterface solver) {
		NeurosomeInterface newSolver = solver.concat(ind);
		System.out.println("\r\n>>Concatenating parent: "+solver.getRepresentation()+"\r\n>>With child: "+ind.getRepresentation()+"\r\n>>Producing: "+newSolver.getRepresentation());
		boolean isBetter = xferTests(newSolver, solver);
		if(isBetter) {
			System.out.println("Which rose above improvement threshold of "+improvementThreshold);
			if(LoadProperties.bstoreImprovedOffspring) {
				Storage.storeSolver(newSolver, LoadProperties.sxferDb);
				System.out.println("*** transfer data stored.***");
			}
		} else {
			System.out.println("Which fell below improvement threshold of "+improvementThreshold);
		}
		return isBetter;
	}
	
	/**
	 * Compares stored solver with passed presumed best individual against dataset input test vectors
	 * to determine if new best individual exceeds accuracy of stored solver by given percentage.
	 * @param nt The new concatenated Neurosome from Solver + best fit
	 * @param solver The original parent solver
	 * @return true if passed Neurosome is more accurate by given threshold. 
	 */
	public boolean xferTests(NeurosomeInterface nt, NeurosomeInterface solver)  {
		int nInErr = 0;
		int oInErr = 0;
		int totalo = 0;
		int totaln = 0;
		//NeuralNet.SHOWEIGHTS = true;
		//System.out.println("Neurosome original: "+(solver == null ? "NULL" : solver.getRepresentation()));
		//System.out.println("Neurosome concat: "+(nt == null? "NULL" : nt.getRepresentation()));
		//System.out.println("world "+(getWorld() == null ? "WORLD NULL" : getWorld()));
    	for(int step = 0; step < imageVecs.size(); step++) {
			ArrayList<double[]> outNeuro1 = solver.execute(imageVecs.get(step));
			String[] imageLabels2 = imageLabels.get(step);
			//System.out.println("Input "+imageLabels[step]+" Output1:"+Arrays.toString(outNeuro1));
			// chain the output
			//double[] outNeuro = nt.execute(outNeuro1);
			for(int step2 = 0; step2 < outNeuro1.size(); step2++) {
				double[] outNeuro2 = outNeuro1.get(step2);
				String opredicted = classify(outNeuro2);
				if (!opredicted.equals(imageLabels2[step2])) {
					++oInErr;
				}
				++totalo;
			}
			//System.out.printf("Predicted: %s\t\tActual:%s cat=%d File:%s\n", opredicted, imageLabels[step],categoryNames.indexOf(imageLabels[step]), imageFiles[step]);
			ArrayList<double[]> outNeuro = nt.execute(imageVecs.get(step));
			//System.out.println("Input "+imageLabels[step]+" Output2:"+Arrays.toString(outNeuro));	
			for(int step2 = 0; step2 < outNeuro.size(); step2++) {
				double[] outNeuro2 = outNeuro.get(step2);
				String predicted = classify(outNeuro2);
				if (!predicted.equals(imageLabels2[step2])) {
					++nInErr;
				}
				++totaln;
			}
			//System.out.printf("Predicted: %s\t\tActual:%s cat=%d File:%s\n", predicted, imageLabels[step],categoryNames.indexOf(imageLabels[step]), imageFiles[step]);
		}	
		double oaccuracy = ((double) (totalo - oInErr)) / (double)totalo;
		double naccuracy = ((double) (totaln - nInErr)) / (double)totaln;
		double improvement = naccuracy - oaccuracy;
		System.out.println("Improvement of best = "+improvement);
		return (improvement >= improvementThreshold);
	}
	
}

