package com.enderling.mesh;

import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;

/** Lazy loader/cache for the four enderling meshes (near + far LOD). No Minecraft dependencies. */
public final class Meshes {
	public static final String[] NAMES = {"main", "bruiser", "lanky", "ember"};
	private static final MeshData[] NEAR = new MeshData[NAMES.length];
	private static final MeshData[] FAR = new MeshData[NAMES.length];

	private Meshes() {}

	public static synchronized MeshData get(int variant, boolean far) {
		MeshData[] cache = far ? FAR : NEAR;
		if (cache[variant] == null) {
			String path = "/assets/enderling/meshes/" + NAMES[variant] + (far ? "_lod" : "") + ".bin";
			try (InputStream in = Meshes.class.getResourceAsStream(path)) {
				if (in == null) throw new IOException("missing resource " + path);
				cache[variant] = MeshData.load(in);
			} catch (IOException e) {
				throw new UncheckedIOException(e);
			}
		}
		return cache[variant];
	}
}
