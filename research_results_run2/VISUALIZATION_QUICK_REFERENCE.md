# Visualization Quick Reference Guide

**Quick access guide to all visualizations**

---

## 🎯 Need a Specific Type of Visualization?

### Architecture Diagrams
📁 `comprehensive_visualizations/1_model_architectures/`

- **MLP architecture?** → `mlp_architecture.png`
- **DeepONet architecture?** → `deeponet_architecture.png`
- **Koopman framework?** → `koopman_operator_diagram.png`

### Orbit & Trajectory Analysis
📁 `comprehensive_visualizations/2_koopman_orbits/`

- **Single/multi-step predictions?** → `real_orbit_predictions.png`
- **Cross-system comparison?** → `orbit_comparison_ifs_systems.png`

### Training & Convergence
📁 `comprehensive_visualizations/3_training_dynamics/`

- **Loss curves?** → `training_curves.png`
- **Convergence analysis?** → `convergence_analysis.png`

### Spectral Properties
📁 `comprehensive_visualizations/4_spectral_analysis/`

- **Eigenvalue spectra?** → `eigenvalue_analysis.png`
- **Stability analysis?** → `spectral_properties.png`

### Error & Performance
📁 `comprehensive_visualizations/5_error_analysis/`

- **Error distributions?** → `error_distributions.png`
- **Spatial error maps?** → `spatial_error_maps.png`
- **Performance heatmaps?** → `performance_heatmaps.png`
- **Residual analysis?** → `residual_analysis.png`
- **Overall comparison?** → `comparison_summary.png`

---

## 📄 Need Documentation?

- **Comprehensive index:** `comprehensive_visualizations/INDEX.md`
- **Summary document:** `COMPREHENSIVE_VISUALIZATIONS_SUMMARY.md`
- **This quick reference:** `VISUALIZATION_QUICK_REFERENCE.md`

---

## 🚀 Common Use Cases

### Writing a Paper?
**Methods section:**
1. `koopman_operator_diagram.png` - Framework
2. `mlp_architecture.png` - MLP details
3. `deeponet_architecture.png` - DeepONet details

**Results section:**
1. `orbit_comparison_ifs_systems.png` - Predictions
2. `eigenvalue_analysis.png` - Spectral analysis
3. `performance_heatmaps.png` - Performance
4. `comparison_summary.png` - Overall results

### Making a Presentation?
**Core slides (5-7 figures):**
1. `koopman_operator_diagram.png`
2. `mlp_architecture.png` OR `deeponet_architecture.png`
3. `real_orbit_predictions.png`
4. `eigenvalue_analysis.png`
5. `performance_heatmaps.png`
6. `comparison_summary.png`

### Updating Documentation?
**README:**
- Architecture diagrams
- Orbit examples
- Performance heatmaps

**Technical docs:**
- All architecture diagrams
- Training curves
- Error analysis

---

## 📊 By Research Question

### "How do the architectures work?"
→ `1_model_architectures/` (all 3 files)

### "How accurate are the predictions?"
→ `2_koopman_orbits/` + `5_error_analysis/error_distributions.png`

### "How well do they train?"
→ `3_training_dynamics/` (both files)

### "Are the models stable?"
→ `4_spectral_analysis/` (both files)

### "Which model is better?"
→ `5_error_analysis/comparison_summary.png`

---

## 🎨 By Visual Type

### Diagrams & Schematics
- `mlp_architecture.png`
- `deeponet_architecture.png`
- `koopman_operator_diagram.png`

### Scatter Plots
- `real_orbit_predictions.png`
- `orbit_comparison_ifs_systems.png`
- `convergence_analysis.png`
- `eigenvalue_analysis.png`
- `spatial_error_maps.png`

### Line Plots
- `training_curves.png`
- `real_orbit_predictions.png` (trajectories)

### Heatmaps
- `performance_heatmaps.png`
- `residual_analysis.png` (2D histograms)

### Bar Charts
- `convergence_analysis.png`
- `spectral_properties.png`
- `comparison_summary.png`

### Histograms
- `error_distributions.png`

### Box Plots
- `error_distributions.png`
- `spectral_properties.png`
- `convergence_analysis.png`

---

## 🔍 By System

### Sierpinski Gasket
All figures include Sierpinski data, especially:
- `orbit_comparison_ifs_systems.png`
- `real_orbit_predictions.png`
- `spatial_error_maps.png`

### Barnsley Fern
All figures include Barnsley data, especially:
- `orbit_comparison_ifs_systems.png`

### Julia Set
Included in aggregate analyses:
- `performance_heatmaps.png`
- `comparison_summary.png`
- `eigenvalue_analysis.png`

### All Systems
- `performance_heatmaps.png`
- `comparison_summary.png`
- `eigenvalue_analysis.png`
- `spectral_properties.png`
- `error_distributions.png`

---

## 📏 By Detail Level

### High-Level Overview
- `koopman_operator_diagram.png`
- `comparison_summary.png`
- `performance_heatmaps.png`

### Medium Detail
- `mlp_architecture.png`
- `deeponet_architecture.png`
- `orbit_comparison_ifs_systems.png`
- `eigenvalue_analysis.png`

### Detailed Analysis
- `real_orbit_predictions.png`
- `training_curves.png`
- `convergence_analysis.png`
- `spectral_properties.png`
- `error_distributions.png`
- `spatial_error_maps.png`
- `residual_analysis.png`

---

## 🎯 Quick Stats

- **Total figures:** 14
- **Categories:** 5
- **Resolution:** 600 DPI (all)
- **Format:** PNG (all)
- **Total size:** ~50-100 MB
- **Generation time:** ~2-3 minutes

---

## 🔧 Regeneration

**Regenerate all:**
```bash
python experiments/create_comprehensive_visualizations.py
```

**Location:**
```
research_results_run2/comprehensive_visualizations/
```

---

**Quick Reference Complete**
