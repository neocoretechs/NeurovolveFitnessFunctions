

import java.io.Serializable;

public class StereoscopicImageBytes implements Comparable, Serializable {
	private static final long serialVersionUID = 1L;
	private byte[] left;
	private byte[] right;
	public StereoscopicImageBytes(byte[] bytesl, byte[] bytesr) {
		this.left = bytesl;
		this.right = bytesr;
	}
	/**
	 * @return the right
	 */
	public byte[] getRight() {
		return right;
	}
	/**
	 * @param right the right to set
	 */
	public void setRight(byte[] right) {
		this.right = right;
	}
	/**
	 * @return the left
	 */
	public byte[] getLeft() {
		return left;
	}
	/**
	 * @param left the left to set
	 */
	public void setLeft(byte[] left) {
		this.left = left;
	}
	@Override
	public int compareTo(Object o) {
		if(left.length > ((StereoscopicImageBytes)o).getLeft().length)
			return 1;
		if(left.length < ((StereoscopicImageBytes)o).getLeft().length)
			return -1;
		if(right.length > ((StereoscopicImageBytes)o).getRight().length)
			return 1;
		if(right.length < ((StereoscopicImageBytes)o).getRight().length)
			return -1;
		for(int i = 0; i < left.length; i++) {
			if(left[i] > ((StereoscopicImageBytes)o).getLeft()[i])
				return 1;
			if(left[i] < ((StereoscopicImageBytes)o).getLeft()[i])
				return -1;
		}
		for(int i = 0; i < left.length; i++) {
			if(left[i] > ((StereoscopicImageBytes)o).getLeft()[i])
				return 1;
			if(left[i] < ((StereoscopicImageBytes)o).getLeft()[i])
				return -1;
		}
		for(int i = 0; i < right.length; i++) {
			if(right[i] > ((StereoscopicImageBytes)o).getRight()[i])
				return 1;
			if(right[i] < ((StereoscopicImageBytes)o).getRight()[i])
				return -1;
		}
		return 0;
	}

}
