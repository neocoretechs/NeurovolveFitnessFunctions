
import java.io.IOException;
import java.io.InputStream;
import java.io.Serializable;

public class RawInstance implements Serializable {
	private static final long serialVersionUID = 1L;
	private static boolean DEBUG = true;
	private double[] image;
	private String label;
	private String name;
	private int width, height;

	/** Constructs the Instance from a BufferedImage. */
	public RawInstance(String name, int width, int height, int bufferSize, String label) {
		this.name = name;
		this.label = label;
		this.width = width;
		this.height = height;
		image = new double[bufferSize];
	}
	
	public void readImage(InputStream in) throws IOException {
		int i = 0;
		int f = 0;
		while((f = in.read()) != -1) {
			image[i++] = ((float)f)/255.0f;
		}
		if(DEBUG)
			System.out.println("File:"+name+" read "+i+" bytes.");
	}
	public double[] getImage() {
		return image;
	}
	
	public String getName() {
		return name;
	}
	
	/** Gets the image width. */
	public int getWidth() {
		return width;
	}

	/** Gets the image height. */
	public int getHeight() {
		return height;
	}

	/** Gets the image label. */
	public String getLabel() {
		return label;
	}
	
	public String toString() {
		return image.toString()+" "+label;
	}
}
