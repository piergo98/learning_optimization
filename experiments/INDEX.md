# Architecture Comparison Study - File Index

This document provides a complete index of all files created for the architecture comparison study.

## 📂 Directory Structure

```
experiments/
├── architecture_comparison_study.py      # Main comprehensive study script
├── quick_architecture_comparison.py      # Fast exploration script
├── analyze_results.py                    # Post-processing and analysis tool
├── run_example_workflow.py               # Interactive example workflow
├── README.md                             # Detailed usage instructions
├── ARCHITECTURE_STUDY_GUIDE.md           # Comprehensive guide
├── QUICK_REFERENCE.md                    # Quick reference guide
└── INDEX.md                              # This file
```

## 📄 File Descriptions

### Core Scripts (Python)

#### 1. `architecture_comparison_study.py` (Comprehensive)
**Purpose:** Full-scale architecture comparison study  
**Runtime:** 2-4 hours on GPU  
**Architectures tested:** ~30  
**Output:** Detailed results, plots, saved models  

**Key functions:**
- `generate_architectures()` - Creates architecture variants
- `train_architecture()` - Trains single network
- `evaluate_network_on_test_set()` - Performance evaluation
- `plot_architecture_comparison()` - Visualization
- `compute_optimal_solution()` - Ground truth baseline

**Key classes:**
- `ExperimentConfig` - Configuration parameters

**Usage:**
```bash
python experiments/architecture_comparison_study.py
```

---

#### 2. `quick_architecture_comparison.py` (Fast)
**Purpose:** Rapid architecture exploration  
**Runtime:** 5-15 minutes on GPU  
**Architectures tested:** 7  
**Output:** Summary results and single plot  

**Key functions:**
- `generate_quick_architectures()` - Representative architectures
- `compute_optimal_solution_simple()` - Simplified optimization
- `evaluate_network()` - Quick evaluation
- `plot_quick_results()` - Summary visualization

**Key classes:**
- `QuickConfig` - Reduced parameters for speed

**Usage:**
```bash
python experiments/quick_architecture_comparison.py
```

---

#### 3. `analyze_results.py` (Analysis)
**Purpose:** Post-processing and visualization  
**Input:** CSV files from previous studies  
**Output:** Plots, statistics, LaTeX tables  

**Key functions:**
- `load_results()` - Load study results
- `print_summary_statistics()` - Statistical summary
- `print_top_architectures()` - Best performers
- `plot_correlation_matrix()` - Metric correlations
- `plot_depth_vs_width_analysis()` - Architecture effects
- `compare_studies()` - Multi-run comparison
- `export_latex_table()` - Publication tables

**Command-line options:**
```bash
# Analyze quick study
python experiments/analyze_results.py --study quick

# Analyze comprehensive study with plots
python experiments/analyze_results.py --study comprehensive --save-plots

# Compare multiple runs
python experiments/analyze_results.py --compare run1.csv run2.csv

# Export LaTeX table
python experiments/analyze_results.py --study comprehensive --export-latex table.tex
```

---

#### 4. `run_example_workflow.py` (Tutorial)
**Purpose:** Interactive tutorial workflow  
**Runtime:** Depends on user choices  
**Output:** Guides through complete study  

**Key functions:**
- `run_quick_study()` - Execute quick study
- `analyze_quick_results()` - Show results
- `run_comprehensive_study()` - Optional full study
- `demonstrate_custom_analysis()` - Custom plots

**Usage:**
```bash
python experiments/run_example_workflow.py
```

---

### Documentation (Markdown)

#### 1. `README.md`
**Content:**
- Installation and setup
- Detailed usage instructions
- Output structure
- Results interpretation
- Extending the study
- Troubleshooting

**When to use:** First-time setup and detailed reference

---

#### 2. `ARCHITECTURE_STUDY_GUIDE.md`
**Content:**
- Comparison of quick vs comprehensive studies
- Understanding metrics
- Example workflows
- Customization guide
- Performance tips

**When to use:** Planning your study and understanding options

---

#### 3. `QUICK_REFERENCE.md`
**Content:**
- Quick start commands
- Common customizations
- Typical workflow
- Results interpretation
- Example analysis scripts
- Complete checklist

**When to use:** Daily reference while running studies

---

#### 4. `INDEX.md` (This file)
**Content:**
- Complete file listing
- Descriptions and purposes
- Cross-references
- Navigation guide

**When to use:** Finding specific functionality

---

## 🔗 Cross-Reference Guide

### I want to...

**...run a quick test**
→ Use `quick_architecture_comparison.py`
→ See `QUICK_REFERENCE.md` for quick start

**...run a complete analysis**
→ Use `architecture_comparison_study.py`
→ See `ARCHITECTURE_STUDY_GUIDE.md` for planning

**...understand the results**
→ Use `analyze_results.py`
→ See `README.md` for metrics explanation

**...learn interactively**
→ Use `run_example_workflow.py`
→ Follow the prompts

**...customize architectures**
→ Edit `generate_architectures()` in main scripts
→ See `README.md` customization section

**...compare multiple runs**
→ Use `analyze_results.py --compare`
→ See analysis tool documentation

**...export for publication**
→ Use `analyze_results.py --export-latex`
→ Plots saved automatically

**...troubleshoot issues**
→ See `README.md` troubleshooting section
→ Check `QUICK_REFERENCE.md` for common fixes

---

## 📊 Output Files Generated

### After Quick Study

```
experiments/quick_study/
├── results/
│   └── quick_results.csv              # Performance metrics
└── plots/
    └── quick_comparison.png            # Summary visualization
```

### After Comprehensive Study

```
experiments/architecture_study/
├── results/
│   ├── architecture_comparison_results.csv    # Main results
│   ├── architecture_comparison_results.json   # Detailed JSON
│   └── [arch_name]_[timestamp]/               # Per-architecture
│       └── history.json                        # Training history
├── models/
│   └── [arch_name]_[timestamp].pt             # Saved weights
└── plots/
    ├── architecture_comparison.png             # 6-panel comparison
    ├── architecture_comparison.pdf             # PDF version
    ├── detailed_comparison.png                 # Bar charts
    └── pareto_front.png                        # Efficiency plots
```

### After Analysis

```
experiments/analysis_plots/
├── correlation_matrix.png              # Metric correlations
└── depth_width_analysis.png            # Architecture effects
```

---

## 🎯 Quick Navigation

| Task | Primary File | Reference Docs |
|------|-------------|----------------|
| Quick test | `quick_architecture_comparison.py` | `QUICK_REFERENCE.md` |
| Full study | `architecture_comparison_study.py` | `ARCHITECTURE_STUDY_GUIDE.md` |
| Post-analysis | `analyze_results.py` | `README.md` |
| Learning | `run_example_workflow.py` | All docs |
| Setup help | - | `README.md` |
| Customization | Scripts (edit functions) | `README.md` → Extending |
| Troubleshooting | - | `README.md` → Troubleshooting |

---

## 🔄 Typical Usage Flow

1. **First Time:**
   ```
   Read: QUICK_REFERENCE.md
   Run: run_example_workflow.py
   ```

2. **Quick Test:**
   ```
   Run: quick_architecture_comparison.py
   Analyze: analyze_results.py --study quick
   ```

3. **Full Study:**
   ```
   Run: architecture_comparison_study.py
   Analyze: analyze_results.py --study comprehensive --save-plots
   Export: analyze_results.py --export-latex table.tex
   ```

4. **Iteration:**
   ```
   Modify: generate_architectures() in scripts
   Re-run: chosen study script
   Compare: analyze_results.py --compare old.csv new.csv
   ```

---

## 📚 Learning Path

### Beginner
1. Read `QUICK_REFERENCE.md` introduction
2. Run `run_example_workflow.py`
3. Examine generated plots
4. Read `README.md` metrics section

### Intermediate
1. Run `quick_architecture_comparison.py`
2. Modify `generate_quick_architectures()`
3. Use `analyze_results.py` for custom plots
4. Read `ARCHITECTURE_STUDY_GUIDE.md`

### Advanced
1. Run `architecture_comparison_study.py`
2. Customize all aspects (architectures, training, evaluation)
3. Write custom analysis scripts
4. Compare multiple experimental runs
5. Prepare publication materials

---

## 🔧 Maintenance

### Adding New Architectures
**Edit:** `generate_architectures()` in study scripts  
**Reference:** `README.md` → Extending the Study  

### Changing Problem Setup
**Edit:** `ExperimentConfig` or `QuickConfig` classes  
**Reference:** `QUICK_REFERENCE.md` → Customizations  

### Custom Metrics
**Edit:** `evaluate_network_on_test_set()` in study scripts  
**Reference:** `QUICK_REFERENCE.md` → Advanced Usage  

### New Visualizations
**Edit:** Plot functions in study scripts or `analyze_results.py`  
**Reference:** Example code in `run_example_workflow.py`  

---

## 📞 Getting Help

1. **Check documentation:**
   - Start with `QUICK_REFERENCE.md`
   - Detailed info in `README.md`
   - Examples in `ARCHITECTURE_STUDY_GUIDE.md`

2. **Run examples:**
   - `run_example_workflow.py` for interactive guide

3. **Check output:**
   - Look at generated plots
   - Examine CSV files
   - Review training logs

4. **Troubleshoot:**
   - See `README.md` → Troubleshooting
   - Check CUDA/GPU availability
   - Verify dependencies

---

**Version:** 1.0  
**Last Updated:** January 15, 2026  
**Maintainer:** Pietro  
**Location:** `/home/pietro/data-driven/learning_optimization/experiments/`
