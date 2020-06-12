import com.neocoretechs.neurovolve.Neurosome;
import com.neocoretechs.neurovolve.fitnessfunctions.NeurosomeFitnessFunction;
import com.neocoretechs.neurovolve.worlds.RelatrixWorld;
import com.neocoretechs.neurovolve.worlds.World;
import com.neocoretechs.volvex.functions.False;
import com.neocoretechs.volvex.functions.True;

/**
 * Fitness function for the Xor test expressed as a neural network evolving in the Neurovolve framework.
 * Expression of fitness function as chromosome function. When using the chromosome function in this way we nullify the functions and
 * depth and provide the means to store the guid as a simple name for semantic retrieval from deep store.<p/>
 * In the designated world, we instantiate this class and provide access to it through the computeRawFitness method. It is presumed that the
 * return type will always be a float.
 * @author groff
 *
 */
public class Xor1 extends NeurosomeFitnessFunction {
	private static final long serialVersionUID = 3845408762831534097L;
	// x1, x2 x 4, 2 inputs 4 values to xor
	public final Object[][] seeds = {{new False(),new False()},{new False(),new True()},{new True(),new False()},{new True(), new True()}};
	final float[][] targs = {{0f,.01f},{.99f,1.0f},{.99f,1.0f},{0f,.01f}};
	/**
	 * @param guid
	 */
	public Xor1(World w, String guid) {
		super(w, guid);
	}
	/**
	 * @param argTypes
	 * @param returnType
	 */
	public Xor1(World w) {
		super(w);
	}

	/**
	 * 
	 */
	public Xor1() {}
	    
	    
	    /**
	     *
	     */     
	public Object execute(Neurosome ind) {
		    	 	 float hits = 0;
		             float rawFit = -1;

		             Object[] arg = new Object[1];
		             boolean[][] results = new boolean[(int)((RelatrixWorld)world).MaxSteps][(int) ((RelatrixWorld)world).TestsPerStep];
		            
				     for(int test = 0; test <((RelatrixWorld)world).TestsPerStep ; test++) {
				    	for(int step = 0; step < ((RelatrixWorld)world).MaxSteps; step++) {
				    		float[] res = (float[]) ind.execute(seeds[step]);
				    		if(World.SHOWTRUTH)
				    			System.out.println("ind:"+ind+" seeds["+step+"]="+seeds[step][0]+","+seeds[step][1]+" targs:"+targs[step][0]+","+targs[step][1]+" res:"+res[0]);
				    		if(res[0] >= targs[step][0] && res[0] <= targs[step][1]) {
				    			++hits;
				    			results[step][0] = true;
				    		}
				    	}
				      }
				      
		             //if( al.data.size() == 1 && ((Strings)(al.data.get(0))).data.equals("d")) hits = 10; // test
		             rawFit = (((RelatrixWorld)world).MinRawFitness - hits);
		             // The SHOWTRUTH flag is set on best individual during run. We make sure to 
		             // place the checkAndStore inside the SHOWTRUTH block to ensure we only attempt to process
		             // the best individual, and this is what occurs in the showTruth method
		             ((RelatrixWorld)world).showTruth(ind, rawFit, results);
		             
		             return rawFit;
		     }
		     
		public static float sigDer(float s) {
			return (float) ((float) (1.0/(1.0+Math.exp(-s))) * (1.0 -(1.0/(1.0+Math.exp(-s))))) ; //derivative of sigmoid function
		}

}
