
import java.awt.image.BufferedImage;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.stream.Stream;

import javax.imageio.ImageIO;

import com.neocoretechs.neurovolve.NeurosomeInterface;
import com.neocoretechs.neurovolve.experimental.NeurosomeCPU;
import com.neocoretechs.neurovolve.fitnessfunctions.False;
import com.neocoretechs.neurovolve.fitnessfunctions.NeurosomeFitnessFunction;
import com.neocoretechs.neurovolve.fitnessfunctions.True;
import com.neocoretechs.neurovolve.properties.LoadProperties;
import com.neocoretechs.neurovolve.worlds.RelatrixWorld;
import com.neocoretechs.neurovolve.worlds.World;
import com.neocoretechs.relatrix.client.RelatrixClient;
import com.neocoretechs.relatrix.client.RemoteStream;


//import com.neocoretechs.robocore.machine.bridge.CircularBlockingDeque;

/**
 * Hardwire port 9020 on same reote node as properties file to retrieve images and send to nuerovolve
 * @author Jonathan Groff (C) NeoCoreTechs 2021
 *
 */
public class StereoImage extends NeurosomeFitnessFunction {
	private static final long serialVersionUID = -4154985360521212822L;
	private static boolean DEBUG = false;
	public static RelatrixClient rkvc;
    private static BufferedImage imagel = null;
    private static BufferedImage imager = null;
    private static Object mutex = new Object();

    private static byte[] bufferl = new byte[0];
    private static byte[] bufferr = new byte[0];
    
    double eulers[] = new double[]{0.0,0.0,0.0};

    private static byte[][] bqueue;
	private static int sequenceNumber;
	static long time1;
	
	public final Object[][] seeds = {{new False(),new False()},{new False(),new True()},{new True(),new False()},{new True(), new True()}};
	final float[][] targs = {{0f,.01f},{.99f,1.0f},{.99f,1.0f},{0f,.01f}};
	
	/**
	 * @param guid
	 */
	public StereoImage(World w) {
		super(w);
	}


	public StereoImage() {}
	    	
	@Override
	public Object execute(NeurosomeInterface ind) {
		try {
			rkvc = new RelatrixClient(LoadProperties.slocallIP, LoadProperties.sremoteIp, 9020);
		} catch (IOException e2) {
			throw new RuntimeException();
		}

		try {		
		    Stream stream = rkvc.findStream('?', '?', '?');
			stream.forEach(e -> {
					StereoscopicImageBytes sib = (StereoscopicImageBytes) ((Comparable[]) e)[2];
					synchronized(mutex) {
						bufferl = sib.getLeft(); // 3 byte BGR
						bufferr = sib.getRight(); // 3 byte BGR	
							try {
								InputStream in = new ByteArrayInputStream(bufferl);
								imagel = ImageIO.read(in);
								in.close();
							} catch (IOException e1) {
								System.out.println("Could not convert LEFT image payload due to:"+e1.getMessage());
								return;
							}
							try {
								InputStream in = new ByteArrayInputStream(bufferr);
								imager = ImageIO.read(in);
								in.close();
							} catch (IOException e1) {
								System.out.println("Could not convert RIGHT image payload due to:"+e1.getMessage());
								return;
							}
						}
					++sequenceNumber; // we want to inc seq regardless to see how many we drop	
				//}
		});
		System.out.println("End of retrieval");
		} catch(IllegalArgumentException | IOException iae) {
			iae.printStackTrace();
			return null;
		}
	 	 double hits = 0;
         double rawFit = -1;

         Object[] arg = new Object[1];
         boolean[][] results = new boolean[(int)getWorld().MaxSteps][(int)getWorld().TestsPerStep];
        
	     for(int test = 0; test < getWorld().TestsPerStep ; test++) {
	    	for(int step = 0; step < getWorld().MaxSteps; step++) {
	    		float[] res = (float[]) ind.execute(seeds[step]);
	    		if(getWorld().SHOWTRUTH)
	    			System.out.println("ind:"+ind+" seeds["+step+"]="+seeds[step][0]+","+seeds[step][1]+" targs:"+targs[step][0]+","+targs[step][1]+" res:"+res[0]);
	    		if(res[0] >= targs[step][0] && res[0] <= targs[step][1]) {
	    			++hits;
	    			results[step][0] = true;
	    		}
	    	}
	      }
	      
         //if( al.data.size() == 1 && ((Strings)(al.data.get(0))).data.equals("d")) hits = 10; // test
         rawFit = getWorld().MinCost - hits;
         // The SHOWTRUTH flag is set on best individual during run. We make sure to 
         // place the checkAndStore inside the SHOWTRUTH block to ensure we only attempt to process
         // the best individual, and this is what occurs in the showTruth method
         getWorld().showTruth(ind, rawFit, results);
         
         return rawFit;
	}

	
}

