# Avoiding Inverse Crimes in Twin Experiments

## What is an Inverse Crime?

An **inverse crime** (or **inverse problem crime**) occurs when the exact same numerical model is used to:

1. Generate the synthetic "truth" observations
2. Perform the inversion/assimilation to recover the unknown state

This creates an artificially favorable scenario because:

- **Model error is zero by construction** - the forward model perfectly represents the "true" physics
- **Discretization errors cancel out** - same mesh, same numerical scheme
- **Results are overly optimistic** - real-world performance will be worse

To produce meaningful results that reflect real-world performance, twin experiments should introduce some form of model error between the truth generation and the assimilation.

## Strategies for Avoiding the Inverse Crime

The primary approach for this codebase is **physics perturbation**:

1. **Perturbed bathymetry** - add noise to bed elevation to simulate survey error
2. **Perturbed friction** - scale Manning's n coefficient to simulate calibration uncertainty

These are the most meaningful for real-world relevance because:
- Bathymetry measurement error is always present in surveys
- Manning's n is notoriously hard to calibrate
- These represent the dominant sources of model error in practice

---

## Recommended Approach: Start with One, Then Layer

Rather than applying all perturbations at once, a hierarchical approach is recommended.

### Phase 1: Bathymetry Perturbation Only

Start with bathymetry perturbation to verify your DA method handles model error:
- Well-understood effect on dynamics (changes wave speed and volume)
- Easy to interpret results

### Phase 2: Friction Perturbation Only

Test friction perturbation separately:
- Different physical signature (affects momentum dissipation and timing)
- Helps understand sensitivity to each parameter

### Phase 3: Combined Bathymetry + Friction Perturbation

This gives a realistic test with multiple sources of physics uncertainty simultaneously.

### Why Not All at Once Initially?

- **Debugging** - if DA fails, you won't know which error source broke it
- **Interpretation** - hard to attribute performance to specific factors
- **Masking** - one large error could dominate and hide issues with others

---

## Experimental Design Matrix

A recommended experimental matrix for thesis work:

| Experiment | Bathymetry | Friction | Purpose |
|------------|------------|----------|---------|
| Baseline (crime) | Same | Same | Sanity check - verify DA works |
| A | Perturbed | Same | Bathymetry error only |
| B | Same | Perturbed | Friction error only |
| C | Perturbed | Perturbed | Combined physics error (realistic scenario) |

This progression allows you to:
1. Verify the DA system works in the ideal case (Baseline)
2. Understand sensitivity to each parameter individually (A, B)
3. Test robustness to combined model error (C)

---

## Physics Perturbation: Bathymetry

Bathymetry (bed elevation) directly affects water depth where h = η - b (free surface minus bed elevation). Small perturbations in bathymetry create systematic model error.

### Types of Bathymetry Perturbation

1. **Additive smooth noise**: `b_assim = b_true + ε(x,y)`
   - Use a correlated random field (Gaussian process or truncated Fourier series)
   - Correlation length should be physically meaningful (e.g., related to mesh scale)
   - Avoids creating artificial small-scale features

2. **Multiplicative noise**: `b_assim = b_true * (1 + ε(x,y))`
   - Percentage error that scales with depth
   - More realistic for survey errors (deeper areas have larger absolute uncertainty)

3. **Systematic bias**: `b_assim = b_true + constant_offset`
   - Simulates datum error or systematic survey bias
   - Simple but less realistic spatially

### Typical Magnitudes

- Real bathymetric survey error: 0.1-0.5m in shallow water, 1-5% of depth in deeper water
- For twin experiments: start with 1-5% relative error

### Caution

Need to ensure `h = η - b` stays positive. Large bathymetry perturbations in shallow areas could cause negative depths.

---

## Problem-Specific Bathymetry Recommendations

Different problems in the codebase have different bathymetry characteristics, which affects the choice of perturbation method.

### TidalProblem (tidal.py)

**Bathymetry characteristics**: Completely flat at 10 meters (`h_b = 10 + x[0] * 0`)

**Recommended perturbation**: **Additive smooth noise**

```
b_assim = 10.0 + ε(x,y)
```

**Rationale**: Since the bathymetry is constant, multiplicative noise would just be uniform scaling (equivalent to changing the base depth). Additive smooth noise introduces actual spatial variation that doesn't exist in the truth, creating a more meaningful test of the DA system.

**Suggested parameters**:
- Correlation length: ~500-1000m (a few mesh cells)
- Amplitude: ~0.5-1.0m (5-10% of the 10m depth)

This creates realistic measurement-like uncertainty and introduces spatial structure the DA must contend with.

### ADCIRCProblem / Shinnecock (shinnecock.py)

**Bathymetry characteristics**: Real variable coastal bathymetry read from `_depth.bp` file, with channels, shallow areas, and wetting/drying zones.

**Recommended perturbation**: **Multiplicative noise**

```
b_assim = b_true * (1 + ε(x,y))
```

**Rationale**:
- **Realistic survey error model** - deeper areas have larger absolute uncertainty, shallow areas smaller
- **Preserves relative structure** - channels stay channels, shoals stay shoals
- **Safer for wetting/drying** - shallow areas get small absolute perturbations, reducing risk of negative depths

**Suggested parameters**:
- Correlation length: ~100-500m (related to survey track spacing)
- Relative error std: 2-5% (i.e., `ε(x,y)` has std of 0.02-0.05)

**Special considerations for wetting/drying**:
- Reduce perturbation magnitude in shallow areas (`depth < 1m`)
- Or clamp bathymetry to ensure `h = η - b > h_min`

### Summary Table

| Problem | Bathymetry Type | Recommended Perturbation | Rationale |
|---------|-----------------|-------------------------|-----------|
| TidalProblem | Flat (10m constant) | Additive smooth noise | Introduces spatial variation; multiplicative would just scale uniformly |
| Shinnecock/ADCIRC | Variable coastal | Multiplicative noise | Realistic survey error model; safer for shallow/WD areas |
| DamProblem | Flat (2m constant) | Additive smooth noise | Same logic as TidalProblem |

---

## Physics Perturbation: Friction

Friction (typically Manning's n) affects momentum dissipation in the shallow water equations.

### Recommended Approach: Uniform Scaling

For consistency and simplicity, **uniform scaling** is recommended for all problems:

```
n_assim = n_true * α    where α ∈ [0.8, 1.2]
```

**Rationale:**
- Simple to implement - just multiply the friction field/constant by a scalar
- Clear interpretation - "friction is 15% higher than truth"
- Works for both constant and spatially varying friction fields
- Good for sensitivity analysis

### Types of Friction Perturbation

1. **Uniform scaling**: `n_assim = n_true * α` where `α ∈ [0.8, 1.2]` (Recommended)
   - Simplest approach
   - Tests sensitivity to friction magnitude

2. **Spatially varying**: `n_assim(x) = n_true(x) * (1 + ε(x))`
   - More realistic - friction varies with substrate
   - More complex to implement (need correlated random field)
   - Save for later if uniform scaling works well

3. **Zonally different**: different multipliers in different regions
   - Mimics having wrong land-use classification

### Typical Magnitudes

- Manning's n uncertainty in practice: ±20-50% is common
- For twin experiments: start with ±10-20%
- Suggested experimental values: α = 0.85, 0.90, 1.10, 1.15

---

## Problem-Specific Friction Recommendations

### TidalProblem (tidal.py)

**Friction characteristics**: Constant Manning's n (`TAU_const = 0.02`)

**Recommended perturbation**: **Uniform scaling**

```python
# In problem setup
TAU_const = 0.02 * alpha  # where alpha = 0.9 or 1.1, etc.
```

### ADCIRCProblem / Shinnecock (shinnecock.py)

**Friction characteristics**: Spatially varying Manning's n read from `_mannings_n.bp`

**Recommended perturbation**: **Uniform scaling**

```python
# After reading mannings_n
mannings_n.x.array[:] *= alpha  # where alpha = 0.9 or 1.1, etc.
```

### Summary Table

| Problem | Friction Type | Recommended Perturbation | Implementation |
|---------|--------------|-------------------------|----------------|
| TidalProblem | Constant (0.02) | Uniform scaling (±10-20%) | `TAU_const = 0.02 * α` |
| Shinnecock/ADCIRC | Spatially varying | Uniform scaling (±10-20%) | `mannings_n.x.array[:] *= α` |
| DamProblem | None | N/A | No friction in dam break |

---

## Interaction Between Bathymetry and Friction

An important question: should you perturb them **independently** or **jointly**?

| Approach | Pros | Cons |
|----------|------|------|
| Independent | Cleaner experiment design, can isolate effects | May miss correlated errors |
| Joint | More realistic | Harder to interpret |

### Recommended Progression

1. **Bathymetry only** - run experiments with varying perturbation magnitudes
2. **Friction only** - same approach
3. **Both together** - final "realistic" scenario

This lets you understand the sensitivity to each parameter before combining them.

### Physical Signatures

Both bathymetry and friction affect the forward model, but they have different signatures:

- **Bathymetry** affects wave speed and volume (through depth changes)
- **Friction** affects momentum dissipation and timing (slows flow)

4D-Var might be able to partially compensate for one type of error by adjusting the initial condition in a way that mimics the other. This is an interesting phenomenon to observe - does the analysis "absorb" model error into the initial condition?

---

## Implementation Considerations

When implementing these perturbations in the twin experiment framework:

1. **Two problem instances** may be needed - one for truth, one for assimilation
2. **Perturbation should be reproducible** - use seeded random number generators
3. **Magnitude should be configurable** - allow experiments with varying error levels
4. **Spatial correlation** should be controllable for smooth random fields (bathymetry)

### Physics Perturbation Implementation

**General approach:**
- Truth run uses true (unperturbed) parameters
- Forward model in cost function uses perturbed parameters
- Compare analysis to truth to evaluate DA performance

**For bathymetry:**
- Modify `create_bathymetry()` method or `h_b` field
- For TidalProblem: add smooth noise field to constant depth
- For Shinnecock: multiply depth field by `(1 + noise_field)`

**For friction:**
- Modify `TAU` constant or Manning's n field
- Apply uniform scaling factor α to the friction values
- `friction_perturbed = friction_true * alpha`

---

## Combined Recommendations Summary

| Problem | Bathymetry Perturbation | Friction Perturbation |
|---------|------------------------|----------------------|
| TidalProblem | Additive smooth noise (0.5-1.0m) | Uniform scaling (±10-20%) |
| Shinnecock/ADCIRC | Multiplicative noise (2-5%) | Uniform scaling (±10-20%) |
| DamProblem | Additive smooth noise (0.1-0.2m) | N/A (frictionless) |
