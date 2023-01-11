
import java.io.File;

import java.util.ArrayList;
import java.util.List;

import com.neocoretechs.neurovolve.NeurosomeInterface;
import com.neocoretechs.neurovolve.activation.SoftMax;
import com.neocoretechs.neurovolve.fitnessfunctions.NeurosomeFitnessFunction;
import com.neocoretechs.neurovolve.worlds.World;

/**
 * Fitness function to evolve image recognizer for imgraw datasets and perform same inference as
 * backpropagated solver control instance.
 * Hardwire port 9010 on same remote node as properties file to retrieve images and send to neurovolve
 * @author Jonathan Groff (C) NeoCoreTechs 2022
 *
 */
public class imgraw extends NeurosomeFitnessFunction {
	private static final long serialVersionUID = -4154985360521212822L;
	private static boolean DEBUG = false;
	private static String prefix = "D:/etc/images/trainset/";
    private static Object mutex = new Object();
    private World world;
    private static float breakOnAccuracyPercentage = .8f; // set to 0 for 100% accuracy expected
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
	
	/**
	 * @param guid
	 */
	public imgraw(World w, String guid) {
		super(w, guid);
		this.world = w;
		init();
	}
	/**
	 * @param argTypes
	 * @param returnType
	 */
	public imgraw(World w) {
		super(w);
		this.world = w;
		init();
	}

	public imgraw() {}
	
	public void init() {

		//if(args.length < 2)
		//	throw new Exception("Usage:java Infer <LocalIP Client> <Remote IpServer> <DB Port> <GUID of Neurosome> <Image file or directory>");
		//new RelatrixClient(args[0], args[1], Integer.parseInt(args[2]));
		if(datasetSize == 0) {
			Dataset dataset = Dataset.loadDataset(new File(prefix), null, false);
			datasetSize = dataset.getSize();
			System.out.printf("Dataset from %s loaded with %d images%n", prefix, datasetSize);
			// Construct a new world to spin up remote connection
			//categoryNames.get(index).getName() is category
			// MinRawFitness is steps * testPerStep args one and two of setStepFactors
			world.setStepFactors((float)datasetSize, 1.0f);
			createImageVecs(world, dataset);
		}
	}
	    	
	@Override
	public Object execute(NeurosomeInterface ind) {
		//Long tim = System.currentTimeMillis();
		//System.out.println("Exec "+Thread.currentThread().getName()+" for ind "+ind.getName());
	 	float hits = 0;
        float rawFit = -1;
        int errCount = 0;

        boolean[][] results = new boolean[(int)world.MaxSteps][(int)world.TestsPerStep];
        
	    for(int test = 0; test < world.TestsPerStep ; test++) {
	    	for(int step = 0; step < world.MaxSteps; step++) {
	    		//System.out.println("Test:"+test+"Step:"+step+" "+ind);
	    		double[] outNeuro = ind.execute(imageVecs[step]);
	    		String predicted = classify(outNeuro);
	    		if (!predicted.equals(imageLabels[step])) {
	    			//if(predicted.equals("N/A"))
	    				//System.out.println("ENCOUNTERED N/A AT INDEX:"+step+" FOR:"+imageLabels[step]);
	    			errCount++;
	    		} else {
	    			++hits;
	    			results[step][test] = true;
	    		}
	    	}
	    }
		if(World.SHOWTRUTH)
			System.out.println("ind:"+ind+" hits:"+hits+" err:"+errCount+" "+(hits/world.MinRawFitness)*100+"%");
         rawFit = world.MinRawFitness - hits;
         // break at predetermined accuracy level? adjust rawfit to 0 on that mark
         // MaxSteps * TestsPerStep is MinRawFitness. hits / MinRawFitness  = percentage passed
         if( breakOnAccuracyPercentage > 0 && (hits/world.MinRawFitness) >= breakOnAccuracyPercentage) {
        	 rawFit = 0;
        	 world.showTruth(ind, rawFit, results);
        	 System.out.println("Fitness function accuracy of "+breakOnAccuracyPercentage*100+"% equaled/surpassed by "+(hits/world.MinRawFitness)*100+"%, adjusted raw fitness to zero.");
         } else {
             world.showTruth(ind, rawFit, results);
         }
     	 //System.out.println("Exit "+Thread.currentThread().getName()+" for ind "+ind.getName()+" in "+(System.currentTimeMillis()-tim));
         return rawFit;
	}

	
	/** Returns the predicted label for the image. */
	public static String classify(double[] dprobs) {
		double maxProb = -1;
		int bestIndex = -1;
		SoftMax sf = new SoftMax(dprobs);
		for (int i = 0; i < dprobs.length; i++) {
			double smax = sf.activate(dprobs[i]);
			if (smax > maxProb) {
				maxProb = smax;
				bestIndex = i;
			}
		}
		if(bestIndex == -1)
			return "N/A";
		return categoryNames.get(bestIndex);
	}
	
	private static void createImageVecs(World world, Dataset dataset) {
	    imageVecs = new double[(int)world.MaxSteps][];
	    imageLabels = new String[(int)world.MaxSteps];
	    List<RawInstance> images = dataset.getImages();
    	for(int step = 0; step < world.MaxSteps; step++) {
    		//System.out.println("Test:"+test+"Step:"+step+" "+ind);
    		RawInstance img = images.get(step);
    		imageLabels[step] = img.getLabel();
    		imageVecs[step] = img.getImage();
    	}
	}
	
}

