#!/usr/bin/env python
"""
Utility functions for managing and recovering explainability analysis results
"""

import pickle
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

def load_intermediate_results(filename):
    """Load intermediate results safely"""
    if os.path.exists(filename):
        try:
            with open(filename, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"❌ Error loading {filename}: {str(e)}")
            return None
    else:
        print(f"⚠️ File {filename} not found")
        return None

def combine_explainability_results():
    """Combine results from different intermediate saves"""
    print("🔄 Combining explainability results from intermediate saves")
    
    # Load all available intermediate results
    xgb_results = load_intermediate_results('dry_bean_xgb_explanations.pkl')
    tabpfn_results = load_intermediate_results('dry_bean_tabpfn_explanations.pkl')
    tabicl_results = load_intermediate_results('dry_bean_tabicl_explanations.pkl')
    minimal_tabicl = load_intermediate_results('tabicl_minimal_results.pkl')
    
    combined_explanations = {}
    
    # Combine XGBoost results
    if xgb_results and 'explainer_state' in xgb_results:
        combined_explanations.update(xgb_results['explainer_state'])
        print("✅ XGBoost results added")
    
    # Combine TabPFN results
    if tabpfn_results and 'explainer_state' in tabpfn_results:
        combined_explanations.update(tabpfn_results['explainer_state'])
        print("✅ TabPFN results added")
    
    # Combine TabICL results
    if tabicl_results and 'explainer_state' in tabicl_results:
        combined_explanations.update(tabicl_results['explainer_state'])
        print("✅ TabICL results added")
    elif minimal_tabicl:
        # Use minimal TabICL results
        combined_explanations['TabICL'] = {
            'minimal_shap': minimal_tabicl,
            'status': 'minimal_analysis_completed'
        }
        print("✅ TabICL minimal results added")
    
    return combined_explanations

def create_feature_importance_summary(combined_explanations, feature_names):
    """Create a comprehensive feature importance summary"""
    print("\n📊 Creating feature importance summary")
    
    importance_data = {}
    
    for model_name, explanations in combined_explanations.items():
        if isinstance(explanations, dict):
            # Built-in feature importance
            if 'feature_importance' in explanations:
                importance_data[f'{model_name}_builtin'] = explanations['feature_importance']['importances']
            
            # Permutation importance
            if 'permutation_importance' in explanations:
                importance_data[f'{model_name}_permutation'] = explanations['permutation_importance']['importances_mean']
            
            # SHAP importance
            if 'shap' in explanations:
                shap_values = explanations['shap']['shap_values']
                if shap_values is not None:
                    shap_importance = np.mean(np.abs(shap_values), axis=0)
                    importance_data[f'{model_name}_shap'] = shap_importance
            
            # Minimal SHAP (for TabICL)
            if 'minimal_shap' in explanations:
                importance_data[f'{model_name}_minimal_shap'] = explanations['minimal_shap']['feature_importance']
    
    if importance_data:
        # Create DataFrame
        df_importance = pd.DataFrame(importance_data, index=feature_names)
        
        # Save to CSV
        df_importance.to_csv('combined_feature_importance.csv')
        print("💾 Combined feature importance saved to 'combined_feature_importance.csv'")
        
        # Create visualization
        plt.figure(figsize=(15, 10))
        df_importance.plot(kind='bar', ax=plt.gca())
        plt.title('Combined Feature Importance Across All Models and Methods')
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('combined_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return df_importance
    
    else:
        print("❌ No feature importance data found")
        return None

def generate_recovery_report():
    """Generate a comprehensive recovery report"""
    print("\n📋 EXPLAINABILITY ANALYSIS RECOVERY REPORT")
    print("="*60)
    
    # Check for available files
    files_to_check = [
        'dry_bean_section2_results.pkl',
        'dry_bean_xgb_explanations.pkl',
        'dry_bean_tabpfn_explanations.pkl',
        'dry_bean_tabicl_explanations.pkl',
        'tabicl_minimal_results.pkl',
        'dry_bean_section3_explainability_memory_safe.pkl'
    ]
    
    print("\n📁 File Availability:")
    available_files = []
    for filename in files_to_check:
        if os.path.exists(filename):
            file_size = os.path.getsize(filename) / (1024*1024)  # MB
            print(f"   ✅ {filename} ({file_size:.2f} MB)")
            available_files.append(filename)
        else:
            print(f"   ❌ {filename} (missing)")
    
    # Try to combine results
    if len(available_files) > 1:
        print("\n🔄 Attempting to combine available results...")
        combined_explanations = combine_explainability_results()
        
        if combined_explanations:
            print(f"\n📊 Successfully combined results from {len(combined_explanations)} models:")
            for model_name, explanations in combined_explanations.items():
                if isinstance(explanations, dict):
                    analyses = []
                    if 'feature_importance' in explanations:
                        analyses.append('Built-in')
                    if 'permutation_importance' in explanations:
                        analyses.append('Permutation')
                    if 'shap' in explanations:
                        analyses.append('SHAP')
                    if 'lime' in explanations:
                        analyses.append('LIME')
                    if 'minimal_shap' in explanations:
                        analyses.append('Minimal SHAP')
                    if 'status' in explanations:
                        analyses.append(f"Status: {explanations['status']}")
                    
                    print(f"   • {model_name}: {', '.join(analyses) if analyses else 'No analyses'}")
            
            # Load feature names for summary
            try:
                with open('dry_bean_section2_results.pkl', 'rb') as f:
                    section2_data = pickle.load(f)
                feature_names = section2_data['feature_names']
                
                # Create feature importance summary
                importance_df = create_feature_importance_summary(combined_explanations, feature_names)
                
                if importance_df is not None:
                    print("\n🎯 Top 5 Most Important Features (Average):")
                    avg_importance = importance_df.mean(axis=1).sort_values(ascending=False)
                    for i, (feature, importance) in enumerate(avg_importance.head(5).items()):
                        print(f"   {i+1}. {feature}: {importance:.4f}")
                
            except Exception as e:
                print(f"❌ Error creating feature importance summary: {str(e)}")
        
        else:
            print("❌ No results could be combined")
    
    else:
        print("\n⚠️ Insufficient files available for combination")
    
    print("\n💡 RECOMMENDATIONS:")
    if 'dry_bean_section2_results.pkl' not in [os.path.basename(f) for f in available_files]:
        print("   • Run Section 2 (Model Training) first")
    
    if len(available_files) < 3:
        print("   • Run the memory-safe explainability script: python section3_explainability_ablation_memory_safe.py")
    
    if 'tabicl_minimal_results.pkl' not in [os.path.basename(f) for f in available_files]:
        print("   • For TabICL specifically, run: python tabicl_memory_safe_analysis.py")
    
    print("\n✅ Recovery report completed")

def clean_intermediate_files():
    """Clean up intermediate files to save space"""
    intermediate_files = [
        'dry_bean_xgb_explanations.pkl',
        'dry_bean_tabpfn_explanations.pkl',
        'dry_bean_tabicl_explanations.pkl'
    ]
    
    print("🧹 Cleaning intermediate files...")
    cleaned = 0
    
    for filename in intermediate_files:
        if os.path.exists(filename):
            try:
                os.remove(filename)
                print(f"   ✅ Removed {filename}")
                cleaned += 1
            except Exception as e:
                print(f"   ❌ Error removing {filename}: {str(e)}")
    
    print(f"🧹 Cleaned {cleaned} intermediate files")

if __name__ == "__main__":
    generate_recovery_report()
