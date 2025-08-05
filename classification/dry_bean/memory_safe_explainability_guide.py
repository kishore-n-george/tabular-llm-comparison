#!/usr/bin/env python
"""
Memory-Safe Explainability Analysis Guide
This script provides guidance on running explainability analysis with memory management
"""

def print_usage_guide():
    """Print comprehensive usage guide"""
    print("🔍 MEMORY-SAFE EXPLAINABILITY ANALYSIS GUIDE")
    print("="*60)
    
    print("\n📋 OVERVIEW:")
    print("This guide helps you run explainability analysis for the dry bean dataset")
    print("with memory management to prevent kernel crashes, especially for TabICL.")
    
    print("\n🎯 PROBLEM SOLVED:")
    print("• Kernel crashes when running SHAP for TabICL due to memory constraints")
    print("• Loss of analysis results when kernel crashes")
    print("• Need to restart entire analysis from beginning")
    
    print("\n💡 SOLUTION FEATURES:")
    print("• Memory-safe SHAP analysis with reduced sample sizes")
    print("• Intermediate result saving after each model")
    print("• Aggressive garbage collection between analyses")
    print("• Recovery utilities to combine partial results")
    
    print("\n📁 FILES CREATED:")
    print("1. section3_explainability_ablation_memory_safe.py - Main memory-safe script")
    print("2. tabicl_memory_safe_analysis.py - Standalone TabICL analysis")
    print("3. explainability_recovery_utils.py - Recovery and combination utilities")
    print("4. memory_safe_explainability_guide.py - This guide")
    
    print("\n🚀 USAGE INSTRUCTIONS:")
    print("\n1. FULL MEMORY-SAFE ANALYSIS:")
    print("   cd dry_bean/")
    print("   python section3_explainability_ablation_memory_safe.py")
    print("   ")
    print("   This will:")
    print("   • Run XGBoost analysis and save results")
    print("   • Run TabPFN analysis and save results")
    print("   • Clear memory aggressively")
    print("   • Attempt TabICL with ultra-conservative settings")
    print("   • Save intermediate results at each step")
    
    print("\n2. TABICL-ONLY ANALYSIS (if main script fails):")
    print("   python tabicl_memory_safe_analysis.py")
    print("   ")
    print("   This will:")
    print("   • Use minimal sample sizes (20 train, 10 test)")
    print("   • Process one sample at a time")
    print("   • Clear memory frequently")
    print("   • Save minimal SHAP results")
    
    print("\n3. RECOVERY AND COMBINATION:")
    print("   python explainability_recovery_utils.py")
    print("   ")
    print("   This will:")
    print("   • Check for all intermediate result files")
    print("   • Combine available results")
    print("   • Create comprehensive feature importance summary")
    print("   • Generate recovery report")
    
    print("\n📊 MEMORY MANAGEMENT STRATEGIES:")
    print("\n• Sample Size Reduction:")
    print("  - XGBoost: 100 samples (normal)")
    print("  - TabPFN: 75 samples (reduced)")
    print("  - TabICL: 50 samples (very reduced)")
    print("  - TabICL minimal: 20 samples (ultra-minimal)")
    
    print("\n• Batch Processing:")
    print("  - SHAP values computed in batches of 10 samples")
    print("  - Memory cleared after each batch")
    print("  - Results combined at the end")
    
    print("\n• Garbage Collection:")
    print("  - Explicit gc.collect() calls")
    print("  - Variable deletion between models")
    print("  - Figure closing after plots")
    
    print("\n🔧 INTERMEDIATE FILES CREATED:")
    print("• dry_bean_xgb_explanations.pkl - XGBoost results")
    print("• dry_bean_tabpfn_explanations.pkl - TabPFN results")
    print("• dry_bean_tabicl_explanations.pkl - TabICL results")
    print("• tabicl_minimal_results.pkl - Minimal TabICL results")
    print("• dry_bean_section3_explainability_memory_safe.pkl - Final combined results")
    
    print("\n📈 OUTPUT FILES:")
    print("• [Model]_feature_importance.png - Feature importance plots")
    print("• [Model]_permutation_importance.png - Permutation importance plots")
    print("• [Model]_shap_summary.png - SHAP summary plots")
    print("• [Model]_shap_bar.png - SHAP bar plots")
    print("• [Model]_lime_explanation_[N].png - LIME explanation plots")
    print("• combined_feature_importance.csv - Combined importance table")
    print("• combined_feature_importance.png - Combined importance plot")
    print("• feature_importance_correlation.png - Method correlation heatmap")
    
    print("\n⚠️ TROUBLESHOOTING:")
    print("\nIf kernel still crashes:")
    print("1. Reduce max_samples further in the scripts")
    print("2. Run models individually using separate scripts")
    print("3. Use only permutation importance (skip SHAP)")
    print("4. Increase frequency of garbage collection")
    
    print("\nIf TabICL fails completely:")
    print("1. Run other models first to get partial results")
    print("2. Use recovery utilities to combine available results")
    print("3. Consider TabICL analysis on a machine with more RAM")
    print("4. Use only built-in feature importance for TabICL")
    
    print("\n✅ SUCCESS INDICATORS:")
    print("• Intermediate .pkl files are created after each model")
    print("• Plots are saved successfully")
    print("• No memory errors in console output")
    print("• Final combined results file is created")
    
    print("\n🎯 EXPECTED RESULTS:")
    print("• Feature importance rankings for each model")
    print("• SHAP values and visualizations")
    print("• Cross-model feature importance comparison")
    print("• Consensus on most important features")
    print("• Memory-safe analysis completion")

def print_quick_start():
    """Print quick start instructions"""
    print("\n🚀 QUICK START:")
    print("="*30)
    print("1. cd dry_bean/")
    print("2. python section3_explainability_ablation_memory_safe.py")
    print("3. If TabICL fails: python tabicl_memory_safe_analysis.py")
    print("4. python explainability_recovery_utils.py")
    print("\n✅ Done! Check the generated plots and CSV files.")

def check_prerequisites():
    """Check if prerequisites are available"""
    print("\n🔍 CHECKING PREREQUISITES:")
    print("="*40)
    
    # Check for required files
    required_files = ['dry_bean_section2_results.pkl']
    missing_files = []
    
    for filename in required_files:
        if os.path.exists(filename):
            print(f"✅ {filename} - Found")
        else:
            print(f"❌ {filename} - Missing")
            missing_files.append(filename)
    
    # Check for required packages
    required_packages = ['numpy', 'pandas', 'matplotlib', 'seaborn', 'shap', 'lime']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} - Installed")
        except ImportError:
            print(f"❌ {package} - Missing")
            missing_packages.append(package)
    
    if missing_files:
        print(f"\n⚠️ Missing files: {', '.join(missing_files)}")
        print("   Run Section 2 (Model Training) first!")
    
    if missing_packages:
        print(f"\n⚠️ Missing packages: {', '.join(missing_packages)}")
        print("   Install with: pip install " + " ".join(missing_packages))
    
    if not missing_files and not missing_packages:
        print("\n✅ All prerequisites satisfied!")
        return True
    else:
        return False

if __name__ == "__main__":
    import os
    
    print_usage_guide()
    print_quick_start()
    
    # Check prerequisites
    if check_prerequisites():
        print("\n🎯 Ready to run memory-safe explainability analysis!")
    else:
        print("\n⚠️ Please resolve prerequisites before running analysis.")
