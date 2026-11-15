package com.enderling.mesh;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;

/** Binary mesh produced by convert.py. No Minecraft dependencies. Units are blocks, +Y up, model front = -Z. */
public final class MeshData {
	public static final class Part {
		public String name;
		public int parent;
		public float[] pivot = new float[3];
		public int vertexCount;
		/** x y z nx ny nz u v per vertex */
		public float[] verts;
		/** quads, 4 indices each (triangles are stored as a,b,c,c) */
		public int[] quads;
	}

	public final Part[] parts;
	public final float height;
	// indices of the animated parts (-1 if absent)
	public final int head, tail, armL, armR, foreL, foreR, legL, legR, shinL, shinR, footL, footR;

	private MeshData(Part[] parts, float height) {
		this.parts = parts;
		this.height = height;
		head = find("head"); tail = find("tail");
		armL = find("arm_L"); armR = find("arm_R");
		foreL = find("forearm_L"); foreR = find("forearm_R");
		legL = find("leg_L"); legR = find("leg_R");
		shinL = find("shin_L"); shinR = find("shin_R");
		footL = find("foot_L"); footR = find("foot_R");
	}

	private int find(String n) {
		for (int i = 0; i < parts.length; i++) if (parts[i].name.equals(n)) return i;
		return -1;
	}

	public static MeshData load(InputStream in) throws IOException {
		ByteBuffer b = ByteBuffer.wrap(in.readAllBytes()).order(ByteOrder.LITTLE_ENDIAN);
		byte[] magic = new byte[4];
		b.get(magic);
		if (!"ENDM".equals(new String(magic, StandardCharsets.US_ASCII))) throw new IOException("bad mesh magic");
		int version = b.getInt();
		if (version != 1) throw new IOException("unsupported mesh version " + version);
		float height = b.getFloat();
		int n = b.getInt();
		Part[] parts = new Part[n];
		for (int i = 0; i < n; i++) {
			Part p = new Part();
			int len = b.get() & 0xFF;
			byte[] nm = new byte[len];
			b.get(nm);
			p.name = new String(nm, StandardCharsets.UTF_8);
			p.parent = b.getInt();
			for (int k = 0; k < 3; k++) p.pivot[k] = b.getFloat();
			p.vertexCount = b.getInt();
			int ni = b.getInt();
			p.verts = new float[p.vertexCount * 8];
			b.asFloatBuffer().get(p.verts);
			b.position(b.position() + p.verts.length * 4);
			p.quads = new int[ni];
			b.asIntBuffer().get(p.quads);
			b.position(b.position() + ni * 4);
			parts[i] = p;
		}
		return new MeshData(parts, height);
	}
}
