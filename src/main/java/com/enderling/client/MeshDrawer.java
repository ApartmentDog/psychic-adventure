package com.enderling.client;

import com.enderling.mesh.MeshData;
import com.mojang.blaze3d.vertex.PoseStack;
import com.mojang.blaze3d.vertex.VertexConsumer;
import net.minecraft.client.renderer.texture.OverlayTexture;

/** Pushes a posed {@link MeshData} into a VertexConsumer as quads. */
final class MeshDrawer {
	private MeshDrawer() {}

	// render thread only; reused to avoid per-frame allocation
	private static float[] scratch = new float[0];

	static void draw(MeshData mesh, float[][] mats, PoseStack.Pose pose, VertexConsumer vc, int light) {
		for (int pi = 0; pi < mesh.parts.length; pi++) {
			MeshData.Part part = mesh.parts[pi];
			final float[] m = mats[pi];
			final int n = part.vertexCount;
			if (scratch.length < n * 6) scratch = new float[n * 6];
			final float[] s = scratch;
			final float[] v = part.verts;
			// transform positions (affine) and normals (rotation only) once per vertex
			for (int i = 0; i < n; i++) {
				int a = i * 8, o = i * 6;
				float x = v[a], y = v[a + 1], z = v[a + 2];
				float nx = v[a + 3], ny = v[a + 4], nz = v[a + 5];
				s[o] = m[0] * x + m[1] * y + m[2] * z + m[3];
				s[o + 1] = m[4] * x + m[5] * y + m[6] * z + m[7];
				s[o + 2] = m[8] * x + m[9] * y + m[10] * z + m[11];
				s[o + 3] = m[0] * nx + m[1] * ny + m[2] * nz;
				s[o + 4] = m[4] * nx + m[5] * ny + m[6] * nz;
				s[o + 5] = m[8] * nx + m[9] * ny + m[10] * nz;
			}
			final int[] q = part.quads;
			for (int k = 0; k < q.length; k++) {
				int i = q[k];
				int o = i * 6, a = i * 8;
				vc.addVertex(pose, s[o], s[o + 1], s[o + 2])
					.setColor(-1)
					.setUv(v[a + 6], v[a + 7])
					.setOverlay(OverlayTexture.NO_OVERLAY)
					.setLight(light)
					.setNormal(pose, s[o + 3], s[o + 4], s[o + 5]);
			}
		}
	}
}
