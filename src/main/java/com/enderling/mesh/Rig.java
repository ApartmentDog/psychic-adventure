package com.enderling.mesh;

/** Procedural skeleton animation. Produces one 3x4 row-major matrix per part. No Minecraft dependencies. */
public final class Rig {
	public static final class Params {
		public float bodyYawDeg;    // entity body yaw (Minecraft convention)
		public float headYawDeg;    // head yaw relative to the body
		public float headPitchDeg;  // positive = looking down
		public float walkPos, walkSpeed, age;
		public float death;         // 0..1 falling over sideways
	}

	private static final float[] I = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};

	public static float[][] pose(MeshData m, Params p) {
		int n = m.parts.length;
		float[] angX = new float[n], angY = new float[n];

		float walk = Math.min(1f, Math.max(0f, p.walkSpeed));
		float swing = (float) Math.cos(p.walkPos * 0.6662f) * walk;
		float sway = (float) Math.sin(p.age * 0.067f) * 0.04f;

		float legL = 0.75f * swing, legR = -legL;
		float armL = -0.6f * swing + sway, armR = 0.6f * swing - sway;

		set(angX, m.legL, legL); set(angX, m.legR, legR);
		set(angX, m.shinL, -(0.06f + 1.1f * Math.max(0f, -legL)));
		set(angX, m.shinR, -(0.06f + 1.1f * Math.max(0f, -legR)));
		if (m.footL >= 0 && m.shinL >= 0) angX[m.footL] = -angX[m.shinL] * 0.6f;
		if (m.footR >= 0 && m.shinR >= 0) angX[m.footR] = -angX[m.shinR] * 0.6f;
		set(angX, m.armL, armL); set(angX, m.armR, armR);
		set(angX, m.foreL, 0.12f + 0.6f * Math.max(0f, armL));
		set(angX, m.foreR, 0.12f + 0.6f * Math.max(0f, armR));

		float hy = clamp(p.headYawDeg, -75f, 75f), hp = clamp(p.headPitchDeg, -60f, 60f);
		if (m.head >= 0) {
			angY[m.head] = -(float) Math.toRadians(hy);
			angX[m.head] = -(float) Math.toRadians(hp);
		}
		if (m.tail >= 0) {
			angY[m.tail] = (float) Math.sin(p.walkPos * 0.3331f) * 0.3f * walk + (float) Math.sin(p.age * 0.05f) * 0.08f;
		}

		// root: face the right way, optionally fall over (roll about the model's Z axis)
		float[] root = mul(rotY((float) Math.toRadians(180.0 - p.bodyYawDeg)), rotZ((float) Math.toRadians(90.0 * p.death)));
		root[7] += 0.6f * p.death; // lift so the fallen body lies on the ground, not in it
		float[][] out = new float[n][];
		for (int i = 0; i < n; i++) {
			MeshData.Part part = m.parts[i];
			float[] base = part.parent < 0 ? root : out[part.parent];
			if (angX[i] == 0f && angY[i] == 0f) { out[i] = base; continue; }
			float[] r = mul(rotY(angY[i]), rotX(angX[i]));
			float[] pv = part.pivot;
			float tx = pv[0] - (r[0] * pv[0] + r[1] * pv[1] + r[2] * pv[2]);
			float ty = pv[1] - (r[4] * pv[0] + r[5] * pv[1] + r[6] * pv[2]);
			float tz = pv[2] - (r[8] * pv[0] + r[9] * pv[1] + r[10] * pv[2]);
			r[3] = tx; r[7] = ty; r[11] = tz;
			out[i] = mul(base, r);
		}
		return out;
	}

	private static void set(float[] a, int i, float v) { if (i >= 0) a[i] = v; }
	private static float clamp(float v, float lo, float hi) { return Math.max(lo, Math.min(hi, v)); }

	static float[] rotX(float a) { float c = (float) Math.cos(a), s = (float) Math.sin(a); return new float[]{1, 0, 0, 0, 0, c, -s, 0, 0, s, c, 0}; }
	static float[] rotY(float a) { float c = (float) Math.cos(a), s = (float) Math.sin(a); return new float[]{c, 0, s, 0, 0, 1, 0, 0, -s, 0, c, 0}; }
	static float[] rotZ(float a) { float c = (float) Math.cos(a), s = (float) Math.sin(a); return new float[]{c, -s, 0, 0, s, c, 0, 0, 0, 0, 1, 0}; }

	/** a * b for 3x4 affine matrices. */
	static float[] mul(float[] a, float[] b) {
		float[] o = new float[12];
		for (int r = 0; r < 3; r++) {
			for (int c = 0; c < 3; c++) {
				o[r * 4 + c] = a[r * 4] * b[c] + a[r * 4 + 1] * b[4 + c] + a[r * 4 + 2] * b[8 + c];
			}
			o[r * 4 + 3] = a[r * 4] * b[3] + a[r * 4 + 1] * b[7] + a[r * 4 + 2] * b[11] + a[r * 4 + 3];
		}
		return o;
	}
}
