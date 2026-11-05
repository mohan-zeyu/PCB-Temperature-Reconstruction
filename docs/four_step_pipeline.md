# PCB infrared reconstruction and forecasting in four steps

$$
L_{\mathrm{meas}}
=
\varepsilon L_b(T_{\mathrm{obj}})
+
(1-\varepsilon)L_b(T_{\mathrm{refl}})
$$

$$
\text{Mask the pixel if}\qquad  T_{\text{expected}}-T
$$

$$
_{\text{IR}}>4\text{ K}
$$

$$
\text{heating}=\text{warming}+\text{cooling}-\text{spreading}
$$



This document explains the current `case10` pipeline in four steps. The goal is to
recover temperatures hidden by low-emissivity component surfaces and then predict how
the reconstructed board temperature changes with time.

The current benchmark uses only the powered part of `case10`: frames 0 through 239,
covering 0 through 1498 seconds. The later cooling part is not used.

## Shared temperature equation

Let

- $T$ be temperature above ambient, in K;
- $t$ be time, in s;
- $a$ be the effective sideways spread rate, in px$^2$/s;
- $b$ be the effective cooling rate, in 1/s; and
- $s$ be the local heating rate, in K/s.

The model is

$$
\frac{dT}{dt}=a\,\operatorname{curv}(T)-bT+s.
$$

On the pixel grid,

$$
\operatorname{curv}(T_{i,j})=
T_{i+1,j}+T_{i-1,j}+T_{i,j+1}+T_{i,j-1}-4T_{i,j}.
$$

This is a thin-plate heat balance:

- $a\,\operatorname{curv}(T)$ moves heat sideways;
- $-bT$ removes heat to the surroundings; and
- $s$ adds heat made on the board.

The physical heat equation has been divided by the unknown areal heat capacity. This
is why $a$, $b$, and $s$ are rates rather than conductivity or power. Temperature
images cannot determine the missing absolute scale, so this project never reports
watts, W/K, or W/(m K).

For `case10`,

$$
b=0.006851\ \mathrm{s}^{-1},
\qquad \frac{1}{b}=145.96\ \mathrm{s}.
$$

This is fitted from the board's warm-up curve,

$$
\overline{T}(t)=T_{\mathrm{settled}}+
\left(T_{\mathrm{start}}-T_{\mathrm{settled}}\right)e^{-bt}.
$$

The direct measurement of $a$ is poorly resolved. It suggests roughly
$0.3$--$0.6$ px$^2$/s, but the reconstruction changes little over a wide range. The
production value is therefore the deliberately conservative choice
$$
a=2.0\ \mathrm{px}^2/\mathrm{s}.
$$

These are effective board-level values. The real PCB is not perfectly uniform, and
the camera pixel size is inferred from the board outline rather than independently
calibrated.

## 1. Find and remove the untrustworthy footprints

Metal lids and connector shields have low emissivity. In this experiment they reflect
cooler surroundings and can read much colder than the nearby board. We detect these
pixels once on reference frame 215 and reuse the same mask for all 240 powered frames.

### 1.1 Find sharp cold seeds

For each pixel, take the 80th percentile in its $9\times9$ neighbourhood. Call this
local upper envelope $U_{i,j}$. A pixel becomes a seed when
$$
U_{i,j}-T_{i,j}
>
\max\!\left(4\ \mathrm{K},\ 0.12\max(U_{i,j},0)\right).
$$

This finds sharp cold edges rather than trying to identify the complete component at
once. A small morphological opening removes isolated detections, enclosed seed holes
are filled, and seed components smaller than 6 pixels are removed.

### 1.2 Grow the mask using a conduction prediction

Starting from the seed mask:

1. Include a 1-pixel temporary halo around the current mask.
2. Hide a band extending another 6 pixels into the apparently good board.
3. Predict the hidden temperatures from data farther outside.
4. Add pixels that are more than 4 K colder than the prediction.
5. Repeat until no new pixels qualify.

The footprint detector predates the fitted `case10` rates and uses a separate screening
length
$$
L_{\mathrm{detect}}=100\ \mathrm{px}.
$$

Its fill equation is

$$
\operatorname{curv}(T_{\mathrm{fill}})
-\frac{T_{\mathrm{fill}}}{L_{\mathrm{detect}}^2}=0,
$$

with measured temperatures fixed outside the hidden region. Thus Step 1 uses the same
screened-plate equation family, but it does **not** use the later `case10` ratio $b/a$.

A frontier pixel is absorbed when

$$
T_{\mathrm{fill}}-T_{\mathrm{measured}}>4\ \mathrm{K}.
$$

The 4 K threshold is above the measured 1.9 K p95 fill error for a surrounded patch.
One-sided regions at the board edge are harder and have a p95 error around 4.2 K, so
the detector also has a maximum-mask guard to prevent uncontrolled growth.

### 1.3 Absorb enclosed pockets

If the growing mask surrounds a connected pocket, the whole pocket is temporarily
hidden and tested. It is absorbed when its median prediction residual is above 4 K:
$$
\operatorname{median}_{(i,j)\in P}
\left(T_{\mathrm{fill}}-T_{\mathrm{measured}}\right)>4\ \mathrm{K}.
$$

### 1.4 Add the final 1-pixel edge

After growth converges, holes are filled and the mask is dilated by 1 pixel. This final
edge removes camera-blurred transition pixels from the trusted data. A 2-pixel final
edge would merge separate blocks without improving the later probes.

**Step 1 result:** 12 connected blocks cover 37.8% of the current board mask. The
deepest hidden point is 36.5 px, or about 19.8 mm, from trusted data.

## 2. Find the number and positions of heat sources

The board is divided into two parts because their measurements mean different things:

1. trusted board, where infrared temperature can be used directly; and
2. masked footprints, where only the surrounding trusted board is used.

### 2A. Heat sources on trusted board

#### 2A.1 Calculate a heating-rate map

Rearrange the shared equation:

$$
s=\frac{dT}{dt}+bT-a\,\operatorname{curv}(T).
$$

For the main **rise** estimate, the time derivative at each interior frame is a centred
difference,
$$
\left.\frac{dT}{dt}\right|_n
\approx
\frac{T_{n+1}-T_{n-1}}{t_{n+1}-t_{n-1}},
$$

and the resulting values of $s$ are averaged over the first 70 powered frames.
Averaging is the least-squares estimate of one constant heating rate per pixel under
constant load.

The raw infrared field has a false temperature step at every metal edge. Therefore
the curvature term is set to zero within 3 pixels of a footprint rim. Otherwise that
false step would create false heat-source peaks.

#### 2A.2 Make an independent onset check

The **onset** estimate fits the warming slope over the first four frames:

$$
s_{\mathrm{onset}}\approx\frac{dT}{dt}.
$$

At the start, warming dominates the cooling and spreading terms. This estimate is
noisier, but it depends much less on the chosen $a$ and fitted $b$, so it provides an
independent check on the rise map.

#### 2A.3 Separate compact peaks from broad background heating

Let $G_{12}(s)$ be a Gaussian blur with width 12 px. Define the compact excess

$$
e=s-G_{12}(s).
$$

A trusted-board hot spot must be a local maximum of $e$, exceed 45% of the strongest
allowed maximum, and own at least 6 connected pixels. The full footprint, a further
3-pixel rim, and the outer 8-pixel image border are excluded.

**Trusted-board result:** one reported hot spot is at approximately
$(17.9,10.3)$ mm. Its onset score is only about 1.1 local standard deviations, so it
should be treated as a weakly confirmed board hot spot rather than hidden truth.

### 2B. Heat sources under masked footprints

No temperature inside a footprint is used. Instead, two trusted rings are sampled
outside each block:

$$
2\le d<5\ \mathrm{px}
\qquad\text{and}\qquad
7\le d<11\ \mathrm{px},
$$

where $d$ is distance from the block. The rings are divided into 24 angular sectors.
For usable sector $k$,

$$
g_k=\operatorname{mean}(T\mid\text{inner},k)
-\operatorname{mean}(T\mid\text{outer},k).
$$

The sector is outward when $g_k>0$. Define

$$
f=\frac{\#\{k:g_k>0\}}{\#\{\text{usable sectors}\}}.
$$

The screen is

$$
\begin{aligned}
f&\ge0.80 &&\Rightarrow \text{heat-source candidate},\\
f&\le0.35 &&\Rightarrow \text{passive candidate},\\
0.35&<f<0.80 &&\Rightarrow \text{mixed}.
\end{aligned}
$$

An isolated strong source normally makes temperature fall in most directions away
from its boundary. A passive block in a background gradient often has heat entering
one side and leaving the other. This is a screening rule, not a theorem: nearby
sources, nonuniform material, or a strong board-wide gradient can defeat it.

**Footprint result:** only block 8 passes. It is the 269 mm$^2$ SoC footprint centred
at approximately $(26.1,32.7)$ mm. All 22 usable sectors point outward and their mean
inner-minus-outer difference is 2.13 K. The 970 mm$^2$ connector bank has 22% outward
sectors and a mean difference of $-0.68$ K, so it is classified as passive.

## 3. Reconstruct every masked temperature

The reconstruction is performed independently for every powered frame. It does not
invent a cold initial state: frame 0 is already about 11.3 K above ambient.

### 3.1 Move the measured transient into an effective source

For each frame, estimate the local warming rate $r=dT/dt$ by a straight-line fit to
nearby frames and their real timestamps. Rearrange the shared equation as

$$
a\,\operatorname{curv}(T)-bT+q=0,
\qquad q=s-r.
$$

This rearrangement is exact within the plate model. Under a footprint, neither $s$ nor
$r$ is reliable, but only their difference $q$ is needed for that frame. Notice that
$q$ is an **effective framewise source term**, not physical power. Even a passive block
can have nonzero or negative $q$ while its temperature is changing.

### 3.2 Fit each block's effect on an outside ring

Hide an 8-pixel ring of trusted board around the union of all footprints. Everything
farther outside remains fixed at its measured temperature.

First compute a block-source-free baseline $T_0$. On the hidden trusted ring it still
uses the measured board value of $q=s-r$; only the unknown block terms are set to zero:

$$
a\,\operatorname{curv}(T_0)-bT_0+q_{\mathrm{ring}}=0.
$$

For each sufficiently large block $B$, choose a fixed unit shape $w_B$ and precompute
its zero-boundary influence field $F_B$:

$$
a\,\operatorname{curv}(F_B)-bF_B+w_B=0,
\qquad F_B=0\ \text{outside the hidden domain}.
$$

The provisional SoC shape is centre-weighted; other block shapes are uniform. These
shapes are normalized so the fitted coefficient remains a mean rate in K/s.

Linearity gives

$$
T_{\mathrm{model}}=T_0+\sum_B q_BF_B.
$$

The block coefficients are the unconstrained least-squares solution

$$
\widehat{\boldsymbol q}
=\underset{\boldsymbol q}{\operatorname{argmin}}
\sum_{(i,j)\in\mathrm{ring}}
\left[
T_0(i,j)+\sum_Bq_BF_B(i,j)-T_{\mathrm{measured}}(i,j)
\right]^2.
$$

The fit uses only temperatures outside the footprints. One block is too small to fit;
it is assigned zero effective source and filled from its boundary.

### 3.3 Reconstruct passive and unconfirmed blocks

Restore the hidden ring as trusted data, so the production solve changes only masked
pixels. Recompute the boundary-driven plate fill and add the fitted block influences.

These blocks use a uniform $w_B$. “Passive” does not force $q_B=0$: the fitted $q_B$
also contains the moved transient $-dT/dt$ and model mismatch. The passive verdict
only says that the block did not show strong outward evidence of being a concentrated
heat-producing component.

### 3.4 Reconstruct the heat-producing SoC lid

The ring fit can estimate the SoC's mean level, but it cannot determine the detailed
temperature shape hidden under the lid. The final solve therefore replaces the
provisional centre bump with a broad non-negative source.

Let $D$ be the SoC footprint plus a 1-pixel transition rim. First solve the source-free
boundary field $B$:

$$
a\,\operatorname{curv}(B)-bB=0
\quad\text{in }D,
\qquad
B=T_{\mathrm{measured}}\quad\text{outside }D.
$$

Then compute a zero-boundary unit-source influence $F$:

$$
a\,\operatorname{curv}(F)-bF+1_{\mathrm{SoC}}=0
\quad\text{in }D,
\qquad
F=0\quad\text{outside }D.
$$

Let $m_{\mathrm{fit}}$ be the SoC mean from the provisional ring fit. The final source
rate is

$$
u=\max\!\left(
0,
\frac{m_{\mathrm{fit}}-\operatorname{mean}_{\mathrm{SoC}}(B)}
{\operatorname{mean}_{\mathrm{SoC}}(F)}
\right),
$$

and the reconstructed lid is

$$
T_{\mathrm{final}}=B+uF.
$$

No curve is pasted and no flat centre is imposed. The broad positive source and the
plate equation produce the plateau and its smooth connection to the measured board.
For `case10`,

$$
L_{\mathrm{spread}}=\sqrt{\frac{a}{b}}
=17.1\ \mathrm{px}\approx9.27\ \mathrm{mm}.
$$

This is comparable to the SoC's roughly 15-pixel centre-to-rim depth, so heat from the
distributed source spreads across much of the lid. The precise two-dimensional
response is not a pure exponential and also depends on the finite boundary.

A positive source is necessary. For positive temperature, the source-free screened
plate equation cannot create a new interior maximum above its boundary. Without an
internal source, a hot chip centre is therefore reconstructed too cold.

On the painted `case17` truth check:

- source-free edge fill: 3.68 K mean absolute error and $-2.10$ K bias;
- uniform source-driven fill: 2.63 K mean absolute error and $+0.01$ K bias; and
- final broad-source lid method: 2.29 K mean absolute error and $+0.37$ K bias.

At `case10` frame 200, the deepest SoC quarter is 41.66 K above ambient with the source
and 29.45 K without it.

### 3.5 Record what remains unknown

Each reconstructed pixel receives a model range based on the held-out-ring error,
distance from trusted data, and disagreement between plausible internal shapes. It is
not a statistical confidence interval. Reconstructed pixels remain estimates and are
downweighted in Step 4.

**Step 3 result:** all 240 powered frames are reconstructed. Directly observed pixels
remain unchanged outside the declared SoC transition rim.

## 4. Predict future temperature fields

The production forecaster is not a neural network. It uses a fixed spatial pattern
library, a 72-dimensional linear plate model, and relaxation toward a settled field
estimated from training frames only.

### 4.1 Build a fixed pattern library

The reconstructed board has about 16,000 pixels, but sideways conduction couples
neighbouring pixels. The library starts with 76 candidate patterns:

- the 64 slowest, smoothest natural plate modes; and
- 12 settled source responses: 3 source geometries at trial spread rates
  $0.5$, $1$, $2$, and $4$ px$^2$/s.

For source shape $w_j$, a settled response at trial spread rate $a_j$ solves

$$
\left(bI-a_jC\right)R_j=w_j,
$$

where $C$ is the discrete curvature operator. Pivoted QR removes numerically redundant
candidates and produces an orthonormal basis
$$
V\in\mathbb{R}^{N\times72},
\qquad V^{\mathsf T}V=I.
$$

No measured temperature amplitudes are used to construct these patterns. Their board
mask, source geometries, and cooling rate do come from the earlier data-derived stages.

### 4.2 Reduce the plate dynamics to 72 coordinates

For a field represented by the library,

$$
c=V^{\mathsf T}T,
\qquad T\approx Vc.
$$

The reduced plate operator is

$$
M=V^{\mathsf T}\left(aC-bI\right)V.
$$

Because the curvature operator cannot create energy and $b>0$, every mode of $M$
decays. The reduced model is therefore stable.

For an unforced deviation and any positive time gap,

$$
c(t+\Delta t)=e^{M\Delta t}c(t).
$$

This is an exact advance of the **reduced 72-dimensional model**. Equal time gaps reuse
the same matrix exponential; no small-step numerical marching is needed.

For completeness, if a constant reduced source $g$ acts during the gap,

$$
c(t+\Delta t)=e^{M\Delta t}c(t)
+M^{-1}\left(e^{M\Delta t}-I\right)g.
$$

### 4.3 Estimate the settled field without using validation or test data

The settled field is the mean of the last training block only:

$$
S=\frac{1}{42}\sum_{n=147}^{188}T_n.
$$

Frames 189 onward do not contribute to $S$.

### 4.4 Start exactly from any supplied frame

For a chosen reconstructed starting field $T_{\mathrm{start}}$, define

$$
c_0=V^{\mathsf T}T_{\mathrm{start}},
\qquad
c_S=V^{\mathsf T}S,
$$

and the resolved deviation

$$
d_0=c_0-c_S.
$$

The part of the **deviation from the settled field** that the library cannot represent
is

$$
R_0=(T_{\mathrm{start}}-S)-Vd_0.
$$

This settled-relative definition is important. Using
$T_{\mathrm{start}}-Vc_0$ instead would not reproduce the starting frame exactly when
$S$ has structure outside the 72-pattern basis.

### 4.5 Forecast by relaxing deviations toward the settled field

The production forecast is

$$
T(t)=S+Ve^{Mt}d_0+e^{-kt}R_0,
$$

with the selected residual rate

$$
k=b=0.006851\ \mathrm{s}^{-1}.
$$

This gives

$$
T(0)=S+Vd_0+R_0=T_{\mathrm{start}},
$$

and

$$
\lim_{t\to\infty}T(t)=S.
$$

The first decaying term follows the reduced plate physics. The second is an explicit
closure for small off-basis details; it is not hidden inside a fitted heat source.

### 4.6 Test the source-driven alternative

The rejected alternative used three non-negative, smooth source histories with 12 time
knots. In reduced coordinates it had the form

$$
\frac{dc}{dt}=Mc+D\,u(t),
\qquad u_j(t)\ge0,
$$

where $D$ maps the three source shapes into the pattern coordinates. Even with temporal
smoothness and a small fixed texture correction, this model scored 2.145 K weighted
test error, worse than the 0.270 K persistence baseline. It is kept as an ablation and
is not the production forecast.

### 4.7 Freeze the data split and score forward prediction

The split uses contiguous blocks with three-frame guard gaps:

| Role | Inclusive frames |
|---|---|
| Training | 0--44, 51--92, 147--188 |
| Validation | 99--140 |
| Test | 195--239 |

Each validation or test window supplies its first frame as the known starting state.
Only later frames are scored.

For measurement resolution $r=0.1$ K and Stage 2 model range $U$, the pixel weight is

$$
w=
\begin{cases}
1, & \text{directly measured pixel},\\[4pt]
\displaystyle
\frac{r^2}{r^2+\max(U,r)^2}, & \text{reconstructed pixel}.
\end{cases}
$$

The weighted mean absolute error is

$$
\operatorname{MAE}_w=
\frac{\sum w\,|T_{\mathrm{predicted}}-T_{\mathrm{target}}|}
{\sum w}.
$$

For reconstructed pixels, $T_{\mathrm{target}}$ is the Stage 2 estimate, not a hidden
measurement.

The current results are:

| Forecast | Weighted error |
|---|---:|
| Production validation | 0.185 K |
| Production held-out test | 0.191 K |
| Persistence held-out test | 0.270 K |
| Constant-source ablation | 1.774 K |
| Smooth-source-history ablation | 2.145 K |
| No-source ablation | 15.336 K |
| Frame-0 forecast over all powered frames | 0.422 K |

The production rates were selected using a declared 0.01 K practical parsimony band
around the best validation score. Inside that band, the independently obtained
$a=2.0$ px$^2$/s and $b=0.006851$/s values were retained.

**Step 4 result:** a forecast can start from any reconstructed frame from 0 through
239. With the default 1-second spacing it writes 674 fields, from elapsed time 0 through
673 s, by which point every declared deviation is below 1% of its starting magnitude.
Ambient temperature is held at the starting-frame value as an external scenario; the
model does not predict ambient conditions.

## Final picture

The four steps have different jobs:

1. **Footprint detection** decides which infrared readings cannot be trusted.
2. **Heat-source detection** finds compact heating on trusted board and screens whole
   footprints using only their surroundings.
3. **Reconstruction** fits the outward temperature effect of every large block and
   solves the plate equation to replace all hidden pixels, using a broad positive
   source for the SoC lid.
4. **Forecasting** compresses each field into stable physical patterns and decays its
   deviation toward a training-only settled field while preserving the supplied start
   exactly.

The reconstructed lid temperatures and their forecast errors remain model estimates.
The pipeline produces complete temperature fields, but it does not turn infrared data
into absolute conductivity or power.
