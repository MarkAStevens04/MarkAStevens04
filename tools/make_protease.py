"""
Generate an animated, depth-sorted SVG of HIV-1 protease (PDB 1HSG, homodimer
+ bound inhibitor indinavir/MK1) for the GitHub README.

- Real atomic coordinates -> genuine 3D tumble rendered as a flipbook of frames.
- Protein backbone: depth-shaded sphere impostors forming a volumetric tube.
- Indinavir: half-coloured STICK model (bonds inferred from atom distances).
- Transparent background; SVG-native filter does chromatic aberration + glow.

Also renders a PIL/numpy preview PNG (on split dark/light panels) for iteration.
"""
import math, os, json, numpy as np

# ===========================================================================
#  EDIT ME  ·  camera (open tools/orient.html to dial this in by eye, then
#  paste the numbers back here and re-run:  python tools/make_protease.py)
# ---------------------------------------------------------------------------
#  POSE  ·  where the orbit starts / how the molecule sits
START_YAW_DEG   = 73
START_PITCH_DEG = 242
START_ROLL_DEG  = 0
#  PATH  ·  the orbit the animation sweeps (around a fixed on-screen axis)
ORBIT_AXIS_DEG  = 0      # tilt of the spin axis on screen: 0 = upright spin,
                         # 90 = top-over-bottom tumble, in between = diagonal
SPIN_DIRECTION  = +1     # +1 or -1 to reverse the orbit
# ===========================================================================

PDB = "tools/1hsg.pdb"
OUT_SVG = "assets/hiv-protease.svg"
OUT_WEBP = "assets/hiv-protease.webp"   # raster hero: effects baked into pixels
WEBP_QUALITY = 88                       # lossy WebP w/ alpha; 80-92 is the sweet spot
PREVIEW = os.environ.get("PREVIEW_PNG", "preview.png")

# ---- look & feel ----------------------------------------------------------
W = H = 720
N_FRAMES = 60             # doubled for a smoother spin (125ms/frame over the same turn)
TURN_SECONDS = 7.5
SUBDIV = 3                 # backbone points between consecutive CA atoms
FOCAL = 130.0             # perspective strength
PX_PER_ANG = 8.3          # world->screen scale
TRANSPARENT_BG = True     # README hero is transparent so it sits on any theme
SHOW_CAPTION = False        # the little "HIV-1 PROTEASE / PDB 1HSG" label, bottom-left
SHADE_BG = (10, 13, 26)   # colour the depth-fog tends toward (shading only)
CROP_VERTICAL = True      # trim empty space above/below so it's not so tall
CROP_PAD = 16             # px of breathing room kept around the molecule

STICK_RAD_ANG = 0.34      # indinavir bond radius (Angstrom) -> stick thickness
BOND_MAX_ANG = 1.85       # heavy-atom distance below which atoms are "bonded"

# ---- raster post-fx (baked into the WebP) ---------------------------------
BLOOM_BLUR   = 0.0        # px radius of the soft glow (was 6.0); 0 = crisp, no bloom
CHROMA_SHIFT = 0.0        # px of R/B channel split (was 2.6); 0 = no chromatic aberration

CHAIN_COL = {"A": (40, 224, 208), "B": (176, 108, 255)}   # teal / violet
ELEM_COL = {"C": (228, 232, 240), "N": (91, 140, 255),
            "O": (255, 91, 110), "S": (240, 220, 120),
            "P": (255, 150, 80), "X": (235, 235, 245)}     # stick CPK-ish colours

# ---- parse PDB ------------------------------------------------------------
def parse():
    chains = {"A": [], "B": []}
    lig = []
    for ln in open(PDB):
        rec = ln[:6].strip()
        if rec == "ATOM" and ln[12:16].strip() == "CA":
            ch = ln[21]
            if ch in chains:
                res = int(ln[22:26])
                xyz = (float(ln[30:38]), float(ln[38:46]), float(ln[46:54]))
                chains[ch].append((res, xyz))
        elif rec == "HETATM" and ln[17:20].strip() == "MK1":
            el = (ln[76:78].strip() or ln[12:16].strip()[0]).upper()
            if el == "H":
                continue
            xyz = (float(ln[30:38]), float(ln[38:46]), float(ln[46:54]))
            lig.append((el, np.array(xyz, float)))
    for ch in chains:
        chains[ch].sort(key=lambda t: t[0])
    return chains, lig

def build_scene():
    """Backbone sphere points + ligand atoms + inferred ligand bonds, centered."""
    chains, lig = parse()
    bb = []
    for ch, arr in chains.items():
        col = CHAIN_COL[ch]
        coords = [np.array(xyz, float) for _, xyz in arr]
        for i in range(len(coords)):
            bb.append(dict(pos=coords[i], col=col, r0=1.50))
            if i + 1 < len(coords):                  # smooth the tube
                a, b = coords[i], coords[i + 1]
                for k in range(1, SUBDIV):
                    t = k / SUBDIV
                    bb.append(dict(pos=a * (1 - t) + b * t, col=col, r0=1.40))

    lig_atoms = [dict(pos=p, el=el, col=ELEM_COL.get(el, ELEM_COL["X"]))
                 for el, p in lig]
    # infer bonds from interatomic distance
    bonds = []
    for i in range(len(lig_atoms)):
        for j in range(i + 1, len(lig_atoms)):
            if np.linalg.norm(lig_atoms[i]["pos"] - lig_atoms[j]["pos"]) < BOND_MAX_ANG:
                bonds.append((i, j))

    center = np.mean([a["pos"] for a in bb], axis=0)
    for a in bb + lig_atoms:
        a["pos"] = a["pos"] - center
    return bb, lig_atoms, bonds

# ---- geometry -------------------------------------------------------------
def rot_x(d):
    a = math.radians(d); c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

def rot_y(d):
    a = math.radians(d); c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

def rot_z(d):
    a = math.radians(d); c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

def rot_axis(u, deg):
    """Rotation by `deg` about unit vector u (Rodrigues)."""
    a = math.radians(deg); c, s = math.cos(a), math.sin(a); C = 1 - c
    x, y, z = u
    return np.array([
        [c + x*x*C,   x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s, c + y*y*C,   y*z*C - x*s],
        [z*x*C - y*s, z*y*C + x*s, c + z*z*C]])

def pose_matrix():
    return rot_z(START_ROLL_DEG) @ rot_x(START_PITCH_DEG) @ rot_y(START_YAW_DEG)

def orbit_axis():
    a = math.radians(ORBIT_AXIS_DEG)         # axis lives in the screen plane
    return (math.sin(a), math.cos(a), 0.0)   # 0 deg -> vertical, 90 -> horizontal

def project(pos, M):
    p = M @ pos
    z = p[2]
    scale = FOCAL / (FOCAL - z)
    return (W / 2 + p[0] * scale * PX_PER_ANG,
            H * 0.47 - p[1] * scale * PX_PER_ANG, z, scale)

def shade(col, t, kind):
    if kind == "lig":
        bright, fog = 0.84 + 0.16 * t, (1 - t) * 0.20
    else:
        bright, fog = 0.42 + 0.58 * t, (1 - t) * 0.33
    return tuple(int(max(0, min(255, c * bright * (1 - fog) + bgc * fog)))
                 for c, bgc in zip(col, SHADE_BG))

def frame_prims(bb, lig_atoms, bonds, spin_deg):
    """Return draw primitives for one frame, sorted far->near."""
    # orbit about the fixed on-screen axis, applied AFTER the starting pose
    M = rot_axis(orbit_axis(), SPIN_DIRECTION * spin_deg) @ pose_matrix()
    allpos = [a["pos"] for a in bb] + [a["pos"] for a in lig_atoms]
    proj = [project(p, M) for p in allpos]
    zs = [p[2] for p in proj]
    zmin, zmax = min(zs), max(zs)
    def tof(z): return (z - zmin) / (zmax - zmin + 1e-9)

    prims = []
    # backbone spheres
    for a, (sx, sy, z, sc) in zip(bb, proj[:len(bb)]):
        t = tof(z)
        r = a["r0"] * PX_PER_ANG * (0.62 + 0.55 * t) * sc
        prims.append(("a", z, sx, sy, r, shade(a["col"], t, "bb"), t, True))
    # ligand: sticks (half-coloured) + small node caps
    lp = proj[len(bb):]
    for i, j in bonds:
        x1, y1, z1, _ = lp[i]; x2, y2, z2, _ = lp[j]
        zm = (z1 + z2) / 2; t = tof(zm)
        w = STICK_RAD_ANG * 2 * PX_PER_ANG * (0.62 + 0.55 * t) * (FOCAL / (FOCAL - zm))
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        prims.append(("l", zm, x1, y1, mx, my, w, shade(lig_atoms[i]["col"], t, "lig")))
        prims.append(("l", zm, mx, my, x2, y2, w, shade(lig_atoms[j]["col"], t, "lig")))
    for a, (sx, sy, z, sc) in zip(lig_atoms, lp):
        t = tof(z)
        r = STICK_RAD_ANG * PX_PER_ANG * (0.62 + 0.55 * t) * sc
        prims.append(("a", z, sx, sy, r, shade(a["col"], t, "lig"), t, False))
    prims.sort(key=lambda p: p[1])
    return prims

# ---- SVG ------------------------------------------------------------------
def hexc(c): return "#%02x%02x%02x" % c

def vertical_extent(frames):
    """Tightest [ymin, ymax] the molecule reaches over the whole spin (+glow)."""
    GLOW = 11  # vertical spread of the blur/bloom filter
    lo, hi = 1e9, -1e9
    for prims in frames:
        for pr in prims:
            if pr[0] == "l":
                _, _z, x1, y1, x2, y2, w, _ = pr
                lo = min(lo, min(y1, y2) - w / 2); hi = max(hi, max(y1, y2) + w / 2)
            else:
                _, _z, sx, sy, r, *_ = pr
                lo = min(lo, sy - r); hi = max(hi, sy + r)
    return lo - GLOW, hi + GLOW

def write_svg(bb, lig_atoms, bonds):
    frames = [frame_prims(bb, lig_atoms, bonds, 360.0 * f / N_FRAMES)
              for f in range(N_FRAMES)]
    if CROP_VERTICAL:
        top, bot = vertical_extent(frames)
        top = max(0.0, top - CROP_PAD); bot = min(float(H), bot + CROP_PAD)
    else:
        top, bot = 0.0, float(H)
    vh = bot - top
    cap_y2 = bot - 12              # caption pinned to the new bottom-left
    cap_y1 = cap_y2 - 20
    L = ['<?xml version="1.0" encoding="UTF-8"?>']
    L.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 {top:.0f} {W} {vh:.0f}" '
             f'width="{W}" height="{vh:.0f}" font-family="ui-monospace,Menlo,Consolas,monospace">')
    L.append('<defs>')
    L.append('<radialGradient id="sph" cx="34%" cy="30%" r="70%">'
             '<stop offset="0%" stop-color="#ffffff" stop-opacity="0.42"/>'
             '<stop offset="42%" stop-color="#ffffff" stop-opacity="0.07"/>'
             '<stop offset="100%" stop-color="#ffffff" stop-opacity="0"/></radialGradient>')
    L.append('''<filter id="fx" x="-20%" y="-20%" width="140%" height="140%" color-interpolation-filters="sRGB">
  <feGaussianBlur in="SourceGraphic" stdDev="5" result="glow"/>
  <feMerge result="base"><feMergeNode in="glow"/><feMergeNode in="SourceGraphic"/></feMerge>
  <feOffset in="base" dx="2.6" dy="0" result="ro"/>
  <feColorMatrix in="ro" type="matrix" values="1 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 1 0" result="rc"/>
  <feOffset in="base" dx="-2.6" dy="0" result="bo"/>
  <feColorMatrix in="bo" type="matrix" values="0 0 0 0 0  0 0 0 0 0  0 0 1 0 0  0 0 0 1 0" result="bc"/>
  <feColorMatrix in="base" type="matrix" values="0 0 0 0 0  0 1 0 0 0  0 0 0 0 0  0 0 0 1 0" result="gc"/>
  <feBlend in="rc" in2="gc" mode="screen" result="rg"/>
  <feBlend in="rg" in2="bc" mode="screen"/>
</filter>''')
    L.append('</defs>')
    if not TRANSPARENT_BG:
        L.append(f'<rect x="0" y="{top:.0f}" width="{W}" height="{vh:.0f}" fill="{hexc(SHADE_BG)}"/>')

    # One filter for the whole flipbook, not one per frame: only a single frame
    # is ever visible (the rest are opacity:0 and contribute nothing), so the
    # browser composites ONE filtered region instead of N_FRAMES of them. This
    # is the difference between the page running smooth and grinding to a halt.
    keytimes = ";".join(f"{i/N_FRAMES:.4f}" for i in range(N_FRAMES))
    L.append('<g filter="url(#fx)">')
    for f, prims in enumerate(frames):
        vals = ";".join("1" if i == f else "0" for i in range(N_FRAMES))
        g = [f'<g opacity="{1 if f==0 else 0}">']
        for pr in prims:
            if pr[0] == "l":
                _, _z, x1, y1, x2, y2, w, col = pr
                if w < 0.5:
                    continue
                g.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                         f'stroke="{hexc(col)}" stroke-width="{w:.1f}" stroke-linecap="round"/>')
            else:
                _, _z, sx, sy, r, col, t, hl = pr
                if r < 0.4:
                    continue
                g.append(f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="{r:.1f}" fill="{hexc(col)}"/>')
                if hl and r > 2.0:
                    g.append(f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="{r:.1f}" '
                             f'fill="url(#sph)" opacity="{0.18+0.45*t:.2f}"/>')
        g.append(f'<animate attributeName="opacity" dur="{TURN_SECONDS}s" '
                 f'calcMode="discrete" repeatCount="indefinite" '
                 f'keyTimes="{keytimes}" values="{vals}"/>')
        g.append('</g>')
        L.append("".join(g))
    L.append('</g>')   # close the single shared-filter wrapper

    # caption: dark outline (paint-order=stroke) so it reads on light OR dark themes
    if SHOW_CAPTION:
        L.append(f'<g opacity="0.92" paint-order="stroke" stroke="#0a0e1a" '
                 f'stroke-width="2.6" stroke-linejoin="round" stroke-opacity="0.55">'
                 f'<text x="24" y="{cap_y1:.0f}" font-size="17" letter-spacing="3" '
                 f'fill="#d7e3ff">HIV-1 PROTEASE</text>'
                 f'<text x="24" y="{cap_y2:.0f}" font-size="12" letter-spacing="2" '
                 f'fill="#9fb4d6">PDB 1HSG  ·  homodimer + indinavir</text></g>')
    L.append('</svg>')
    os.makedirs(os.path.dirname(OUT_SVG), exist_ok=True)
    open(OUT_SVG, "w", encoding="utf-8").write("\n".join(L))
    print("wrote", OUT_SVG, "%.0f KB" % (os.path.getsize(OUT_SVG) / 1024))
    return frames, top, bot

# ---- raster rendering (effects baked into pixels) -------------------------
def render_frame_rgba(prims, SS=2):
    """Rasterize ONE frame to a W x H RGBA image with the glow + chromatic
    aberration baked into the pixels -- the exact look the SVG <filter> made,
    but computed once here instead of live in the browser every frame."""
    from PIL import Image, ImageDraw, ImageFilter
    layer = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
    dr = ImageDraw.Draw(layer, "RGBA")
    for pr in prims:
        if pr[0] == "l":
            _, _z, x1, y1, x2, y2, w, col = pr
            dr.line([x1*SS, y1*SS, x2*SS, y2*SS], fill=col + (255,),
                    width=max(1, int(w*SS)), joint="curve")
        else:
            _, _z, sx, sy, r, col, t, hl = pr
            if r < 0.4:
                continue
            x, y, rr = sx*SS, sy*SS, r*SS
            dr.ellipse([x-rr, y-rr, x+rr, y+rr], fill=col + (255,))
            if hl and rr > 2.0*SS:
                a = int((0.18 + 0.42*t) * 255); hr = rr*0.46
                hx, hy = x-rr*0.28, y-rr*0.32
                dr.ellipse([hx-hr, hy-hr, hx+hr, hy+hr], fill=(255, 255, 255, a))
    rgba = np.asarray(layer, np.float32)
    if BLOOM_BLUR > 0:                                  # optional soft glow / bloom
        glow = np.asarray(layer.filter(ImageFilter.GaussianBlur(BLOOM_BLUR*SS)), np.float32)
        a = rgba[:, :, 3:4]/255; ga = glow[:, :, 3:4]/255
        rgb = rgba[:, :, :3]; grgb = glow[:, :, :3]
        out_rgb = 255*(1-(1-rgb/255*a)*(1-grgb/255*ga*0.9))
        out_a = np.clip(a + ga*0.9, 0, 1) * 255
    else:                                               # crisp: straight shapes, no bloom
        out_rgb = rgba[:, :, :3].copy()
        out_a = rgba[:, :, 3:4]
    if CHROMA_SHIFT > 0:                                # optional chromatic aberration
        dx = int(CHROMA_SHIFT*SS)
        out_rgb[:, :, 0] = np.roll(out_rgb[:, :, 0], dx, axis=1)
        out_rgb[:, :, 2] = np.roll(out_rgb[:, :, 2], -dx, axis=1)
    mol = np.concatenate([out_rgb, out_a], axis=2)
    return Image.fromarray(np.clip(mol, 0, 255).astype(np.uint8)).resize((W, H), Image.LANCZOS)

def write_webp(frames, top, bot):
    """Assemble all frames into a looping, transparent animated WebP. The
    browser GPU-composites a single image loop -- no live SVG filtering -- so
    it stays smooth while keeping the glow + chromatic aberration."""
    y0, y1 = int(round(top)), int(round(bot))
    imgs = [render_frame_rgba(prims).crop((0, y0, W, y1)) for prims in frames]
    dur = int(round(1000 * TURN_SECONDS / N_FRAMES))   # ms per frame
    imgs[0].save(OUT_WEBP, format="WebP", save_all=True, append_images=imgs[1:],
                 duration=dur, loop=0, lossless=False, quality=WEBP_QUALITY,
                 method=6, minimize_size=True)
    print("wrote", OUT_WEBP, "%.0f KB" % (os.path.getsize(OUT_WEBP) / 1024),
          f"  {W}x{y1 - y0}  {len(imgs)} frames @ {dur}ms")

# ---- PIL preview over split dark/light panels (to vet transparency) -------
def preview(frames, idx=0):
    from PIL import Image, ImageDraw
    mol_img = render_frame_rgba(frames[idx])
    # composite over split dark(left)/light(right) to mimic GitHub themes
    bg = Image.new("RGB", (W, H), (13, 17, 23))
    ImageDraw.Draw(bg).rectangle([W//2, 0, W, H], fill=(255, 255, 255))
    bg = bg.convert("RGBA"); bg.alpha_composite(mol_img)
    bg.convert("RGB").save(PREVIEW)
    print("wrote", PREVIEW)

# ---- interactive camera studio (writes a standalone HTML you open in a browser) -
ORIENT_HTML = """<!doctype html><html><head><meta charset="utf-8">
<title>HIV-1 protease - camera studio</title><style>
  :root{color-scheme:dark}
  body{margin:0;background:#0b0f1a;color:#cdd9ee;font:14px ui-monospace,Menlo,Consolas,monospace;
       display:flex;gap:22px;flex-wrap:wrap;justify-content:center;align-items:flex-start;padding:24px}
  canvas{background:#0d1117;border-radius:14px;border:1px solid #1e2740;cursor:grab;touch-action:none}
  canvas:active{cursor:grabbing}
  #panel{min-width:312px;max-width:352px}
  h1{font-size:15px;letter-spacing:2px;color:#e7eefb;margin:2px 0 4px}
  p.hint{color:#7d8db0;line-height:1.5;margin:0 0 14px}
  .sep{margin:16px 0 6px;color:#28e0d0;font-size:12px;letter-spacing:2px;
       border-top:1px solid #1e2740;padding-top:13px}
  .row{display:flex;align-items:center;gap:10px;margin:9px 0}
  .row label{width:58px;color:#9fb4d6}
  .row input[type=range]{flex:1;accent-color:#28e0d0}
  .row input[type=number]{width:60px;background:#121a2e;color:#e7eefb;border:1px solid #28324f;
       border-radius:7px;padding:5px 7px;font:inherit}
  pre{background:#121a2e;border:1px solid #28324f;border-radius:10px;padding:14px;color:#bfe9e2;
       white-space:pre;overflow:auto;margin:14px 0 10px}
  button{border:0;border-radius:9px;padding:10px 14px;font:inherit;font-weight:700;cursor:pointer;letter-spacing:.5px}
  button.prim{background:#28e0d0;color:#04201d}
  button.prim:hover{background:#5cf0e3}
  button.ghost{background:#16203a;color:#cdd9ee;font-weight:600}
  button.ghost:hover{background:#1d2b4d}
  .tip{color:#7d8db0;font-size:12px;margin-top:8px;line-height:1.5}
  kbd{background:#1b2540;border:1px solid #2c3a5e;border-radius:5px;padding:1px 5px;color:#cdd9ee}
</style></head><body>
<canvas id="c" width="460" height="460"></canvas>
<div id="panel">
  <h1>CAMERA STUDIO</h1>
  <p class="hint">Drag to set the starting pose (<kbd>Shift</kbd>+drag = roll). Then choose the
     orbit axis and hit <b>Play</b> to preview the path. Paste the numbers into the
     <b>EDIT ME</b> block of <code>make_protease.py</code> and run it once.</p>
  <div class="sep">START POSE</div>
  <div class="row"><label>Yaw</label><input type="range" id="ya" min="0" max="359" step="1">
     <input type="number" id="yn" min="0" max="359"></div>
  <div class="row"><label>Pitch</label><input type="range" id="pa" min="0" max="359" step="1">
     <input type="number" id="pn" min="0" max="359"></div>
  <div class="row"><label>Roll</label><input type="range" id="ra" min="0" max="359" step="1">
     <input type="number" id="rn" min="0" max="359"></div>
  <div class="sep">CAMERA PATH</div>
  <div class="row"><label>Axis</label><input type="range" id="oa" min="0" max="180" step="1">
     <input type="number" id="on" min="0" max="180"></div>
  <div class="row"><label>Dir</label><button id="dir" class="ghost"></button>
     <button id="play" class="ghost" style="flex:1"></button></div>
  <div class="row"><label>Preview</label><input type="range" id="sc" min="0" max="1000" step="1" value="0"></div>
  <pre id="out"></pre>
  <button id="copy" class="prim">Copy values</button>
  <div class="tip">The dashed line is the orbit axis. Preview scrubs one full turn;
     Play loops it at the real animation speed.</div>
</div>
<script>const DATA = __DATA__;</script>
<script>
const C=document.getElementById('c'),X=C.getContext('2d');
const CW=C.width,CH=C.height, W=DATA.W,H=DATA.H,F=DATA.FOCAL,PX=DATA.PX,BG=DATA.SHADE_BG;
let yaw=DATA.yaw%360, pitch=DATA.pitch%360, roll=DATA.roll%360;
let axis=DATA.axis%360, dir=DATA.dir, frac=0, playing=false, phase=0, last=0;
const D2R=Math.PI/180, norm=v=>((v%360)+360)%360;
const mul=(A,B)=>{let C=[[0,0,0],[0,0,0],[0,0,0]];for(let i=0;i<3;i++)for(let j=0;j<3;j++){let s=0;
  for(let k=0;k<3;k++)s+=A[i][k]*B[k][j];C[i][j]=s;}return C;};
const rx=d=>{let a=d*D2R,c=Math.cos(a),s=Math.sin(a);return[[1,0,0],[0,c,-s],[0,s,c]];};
const ry=d=>{let a=d*D2R,c=Math.cos(a),s=Math.sin(a);return[[c,0,s],[0,1,0],[-s,0,c]];};
const rz=d=>{let a=d*D2R,c=Math.cos(a),s=Math.sin(a);return[[c,-s,0],[s,c,0],[0,0,1]];};
const raxis=(u,d)=>{let a=d*D2R,c=Math.cos(a),s=Math.sin(a),K=1-c,x=u[0],y=u[1],z=u[2];
  return[[c+x*x*K,x*y*K-z*s,x*z*K+y*s],[y*x*K+z*s,c+y*y*K,y*z*K-x*s],[z*x*K-y*s,z*y*K+x*s,c+z*z*K]];};
const ap=(M,p)=>[M[0][0]*p[0]+M[0][1]*p[1]+M[0][2]*p[2],
                 M[1][0]*p[0]+M[1][1]*p[1]+M[1][2]*p[2],
                 M[2][0]*p[0]+M[2][1]*p[1]+M[2][2]*p[2]];
function shade(col,t,lig){let b=lig?0.84+0.16*t:0.42+0.58*t,f=(lig?0.20:0.33)*(1-t);
  return col.map((c,i)=>Math.max(0,Math.min(255,c*b*(1-f)+BG[i]*f))|0);}
const rgb=c=>'rgb('+c[0]+','+c[1]+','+c[2]+')';
function render(){
  const ax=axis*D2R, u=[Math.sin(ax),Math.cos(ax),0];
  const Rp=mul(rz(roll),mul(rx(pitch),ry(yaw)));
  const M=mul(raxis(u,dir*360*frac),Rp);
  const pr=[],lp=[]; let zmin=1e9,zmax=-1e9;
  const proj=p=>{const q=ap(M,p),z=q[2],sc=F/(F-z);return[W/2+q[0]*sc*PX,H*0.47-q[1]*sc*PX,z,sc];};
  for(const a of DATA.bb){const q=proj(a);zmin=Math.min(zmin,q[2]);zmax=Math.max(zmax,q[2]);pr.push([a,q]);}
  for(const a of DATA.lig){const q=proj(a);zmin=Math.min(zmin,q[2]);zmax=Math.max(zmax,q[2]);lp.push([a,q]);}
  const tof=z=>(z-zmin)/(zmax-zmin+1e-9);
  const items=[];
  for(const [a,q] of pr){const t=tof(q[2]);
    items.push({z:q[2],type:'c',x:q[0],y:q[1],r:a[3]*PX*(0.62+0.55*t)*q[3],col:shade(a[4],t,false)});}
  for(const [i,j] of DATA.bonds){const A=lp[i][1],B2=lp[j][1],zm=(A[2]+B2[2])/2,t=tof(zm);
    const w=DATA.STICK_RAD*2*PX*(0.62+0.55*t)*(F/(F-zm)),mx=(A[0]+B2[0])/2,my=(A[1]+B2[1])/2;
    items.push({z:zm,type:'l',x1:A[0],y1:A[1],x2:mx,y2:my,w,col:shade(DATA.lig[i][3],t,true)});
    items.push({z:zm,type:'l',x1:mx,y1:my,x2:B2[0],y2:B2[1],w,col:shade(DATA.lig[j][3],t,true)});}
  for(const [a,q] of lp){const t=tof(q[2]);
    items.push({z:q[2],type:'c',x:q[0],y:q[1],r:DATA.STICK_RAD*PX*(0.62+0.55*t)*q[3],col:shade(a[3],t,true)});}
  items.sort((p,q)=>p.z-q.z);
  X.setTransform(1,0,0,1,0,0);X.clearRect(0,0,CW,CH);X.setTransform(CW/W,0,0,CH/H,0,0);
  // orbit-axis indicator (dashed line through the centre)
  const cx=W/2,cy=H*0.47,L=W*0.46,dxs=Math.sin(ax),dys=-Math.cos(ax);
  X.strokeStyle='rgba(120,150,205,0.30)';X.lineWidth=1.6;X.setLineDash([8,8]);
  X.beginPath();X.moveTo(cx-dxs*L,cy-dys*L);X.lineTo(cx+dxs*L,cy+dys*L);X.stroke();X.setLineDash([]);
  for(const it of items){
    if(it.type==='c'){X.beginPath();X.arc(it.x,it.y,Math.max(0.3,it.r),0,7);X.fillStyle=rgb(it.col);X.fill();}
    else{X.strokeStyle=rgb(it.col);X.lineWidth=it.w;X.lineCap='round';
      X.beginPath();X.moveTo(it.x1,it.y1);X.lineTo(it.x2,it.y2);X.stroke();}
  }
  out.textContent=
     'START_YAW_DEG   = '+Math.round(yaw)+'\\n'+
     'START_PITCH_DEG = '+Math.round(pitch)+'\\n'+
     'START_ROLL_DEG  = '+Math.round(roll)+'\\n'+
     'ORBIT_AXIS_DEG  = '+Math.round(axis)+'\\n'+
     'SPIN_DIRECTION  = '+(dir>0?'+1':'-1');
}
const [ya,yn,pa,pn,ra,rn,oa,on,sc,out,dirBtn,playBtn]=
  ['ya','yn','pa','pn','ra','rn','oa','on','sc','out','dir','play'].map(id=>document.getElementById(id));
function syncInputs(){ya.value=yn.value=Math.round(yaw);pa.value=pn.value=Math.round(pitch);
  ra.value=rn.value=Math.round(roll);oa.value=on.value=Math.round(axis);
  dirBtn.textContent=dir>0?'CW \\u21bb':'CCW \\u21ba';
  playBtn.textContent=playing?'\\u23f8 Pause':'\\u25b6 Play orbit';}
function stopPlay(){playing=false;}
function toStart(){stopPlay();frac=0;phase=0;sc.value=0;}      // pose edits jump to frame 0
function loop(ts){if(!playing)return;if(!last)last=ts;const dt=(ts-last)/1000;last=ts;
  phase=(phase+dt/DATA.turn)%1;frac=phase;sc.value=Math.round(frac*1000);render();
  requestAnimationFrame(loop);}
function bindPose(sl,nu,set){[sl,nu].forEach(el=>el.addEventListener('input',()=>{
  set(norm(+el.value));toStart();syncInputs();render();}));}
bindPose(ya,yn,v=>yaw=v);bindPose(pa,pn,v=>pitch=v);bindPose(ra,rn,v=>roll=v);
[oa,on].forEach(el=>el.addEventListener('input',()=>{axis=Math.max(0,Math.min(180,+el.value));
  syncInputs();render();}));
dirBtn.addEventListener('click',()=>{dir=-dir;syncInputs();render();});
playBtn.addEventListener('click',()=>{playing=!playing;syncInputs();
  if(playing){last=0;requestAnimationFrame(loop);}});
sc.addEventListener('input',()=>{stopPlay();frac=(+sc.value)/1000;phase=frac;syncInputs();render();});
let drag=false,lx=0,ly=0;
C.addEventListener('pointerdown',e=>{drag=true;lx=e.clientX;ly=e.clientY;C.setPointerCapture(e.pointerId);});
C.addEventListener('pointermove',e=>{if(!drag)return;const dx=e.clientX-lx,dy=e.clientY-ly;lx=e.clientX;ly=e.clientY;
  if(e.shiftKey){roll=norm(roll+dx*0.6);}else{yaw=norm(yaw+dx*0.6);pitch=norm(pitch+dy*0.6);}
  toStart();syncInputs();render();});
C.addEventListener('pointerup',()=>drag=false);
document.getElementById('copy').addEventListener('click',()=>{
  navigator.clipboard.writeText(out.textContent);
  const b=document.getElementById('copy');b.textContent='Copied \\u2713';
  setTimeout(()=>b.textContent='Copy values',1200);});
syncInputs();render();
</script></body></html>"""

def export_orient_html(bb, lig_atoms, bonds, out="tools/orient.html"):
    def xyz(p): return [round(float(p[0]), 2), round(float(p[1]), 2), round(float(p[2]), 2)]
    data = {
        "W": W, "H": H, "FOCAL": FOCAL, "PX": PX_PER_ANG, "SHADE_BG": list(SHADE_BG),
        "STICK_RAD": STICK_RAD_ANG, "turn": TURN_SECONDS,
        "yaw": START_YAW_DEG % 360, "pitch": START_PITCH_DEG % 360, "roll": START_ROLL_DEG % 360,
        "axis": ORBIT_AXIS_DEG % 360, "dir": 1 if SPIN_DIRECTION >= 0 else -1,
        "bb": [xyz(a["pos"]) + [a["r0"], list(a["col"])] for a in bb],
        "lig": [xyz(a["pos"]) + [list(a["col"])] for a in lig_atoms],
        "bonds": [[i, j] for i, j in bonds],
    }
    open(out, "w", encoding="utf-8").write(ORIENT_HTML.replace("__DATA__", json.dumps(data)))
    print("wrote", out, "-> open in a browser to pick a camera path")

if __name__ == "__main__":
    bb, lig_atoms, bonds = build_scene()
    print(f"backbone spheres: {len(bb)}  ligand atoms: {len(lig_atoms)}  bonds: {len(bonds)}")
    frames, top, bot = write_svg(bb, lig_atoms, bonds)
    write_webp(frames, top, bot)
    preview(frames, idx=0)
    export_orient_html(bb, lig_atoms, bonds)
