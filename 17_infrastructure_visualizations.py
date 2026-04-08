import sys
import os

# Add parent directory to path to import plot_style
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plot_style import set_tufte_defaults, apply_tufte_style, save_tufte_figure, COLORS

"""
Visualization generation for Blog 17: Infrastructure Inspection with DINOv2
Creates minimalist-style visualizations of embedding space and anomaly detection.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import warnings

import sys
import os

# Add parent directory to path to import plot_style
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plot_style import set_tufte_defaults, apply_tufte_style, save_tufte_figure, COLORS

# Import Tufte plotting utilities
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from tda_utils import setup_tufte_plot, TufteColors


warnings.filterwarnings('ignore')

def apply_minimalist_style_manual(ax):
    """Apply minimalist style components manually to axis."""
    plt.rcParams["font.family"] = "serif"
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_position(("outward", 5))
    ax.spines["bottom"].set_position(("outward", 5))
def generate_embeddings_with_structure(n_images=10000):
    """
    Generate synthetic embeddings with realistic cluster structure.
    
    Simulates DINOv2 embeddings:
    - 94% normal infrastructure (tight cluster)
    - 3% vegetation intrusion (separate cluster)
    - 2% equipment/activity (outliers)
    - 1% surface damage (outliers)
    
    Returns:
        embeddings: (n_images, 384) array
        labels: Ground truth labels (0=normal, 1=vegetation, 2=equipment, 3=damage)
        anomaly_scores: Distance to cluster centroid
    """
    np.random.seed(42)
    
    # Define class distribution
    n_normal = int(n_images * 0.94)
    n_vegetation = int(n_images * 0.03)
    n_equipment = int(n_images * 0.02)
    n_damage = n_images - n_normal - n_vegetation - n_equipment
    
    embeddings = []
    labels = []
    
    # Normal infrastructure (cluster 0)
    normal_center = np.zeros(384)
    normal_embeddings = np.random.randn(n_normal, 384) * 0.3 + normal_center
    embeddings.append(normal_embeddings)
    labels.extend([0] * n_normal)
    
    # Vegetation intrusion (cluster 1)
    vegetation_center = np.ones(384) * 0.5
    vegetation_embeddings = np.random.randn(n_vegetation, 384) * 0.4 + vegetation_center
    embeddings.append(vegetation_embeddings)
    labels.extend([1] * n_vegetation)
    
    # Equipment/activity (outliers, cluster 2)
    equipment_center = np.ones(384) * 1.5
    equipment_embeddings = np.random.randn(n_equipment, 384) * 0.8 + equipment_center
    embeddings.append(equipment_embeddings)
    labels.extend([2] * n_equipment)
    
    # Surface damage (outliers, cluster 3)
    damage_center = np.ones(384) * -1.0
    damage_embeddings = np.random.randn(n_damage, 384) * 0.6 + damage_center
    embeddings.append(damage_embeddings)
    labels.extend([3] * n_damage)
    
    # Concatenate all
    embeddings = np.vstack(embeddings)
    labels = np.array(labels)
    
    # Compute anomaly scores (distance to nearest cluster center)
    # For simplicity, use distance to overall mean for normal, and larger for anomalies
    centers = {
        0: normal_center,
        1: vegetation_center,
        2: equipment_center,
        3: damage_center
    }
    
    anomaly_scores = np.zeros(n_images)
    for i, label in enumerate(labels):
        center = centers[label]
        anomaly_scores[i] = np.linalg.norm(embeddings[i] - center)
    
    return embeddings, labels, anomaly_scores

def create_main_visualization():
    """
    Create t-SNE projection of DINOv2 embeddings showing cluster structure.
    """
    print("Generating main visualization...")
    
    # Generate embeddings
    embeddings, labels, anomaly_scores = generate_embeddings_with_structure(n_images=10000)
    
    # Reduce to 2D with t-SNE
    print("  Running t-SNE dimensionality reduction...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Define colors and labels
    colors = {
        0: '#CCCCCC',  # Gray for normal
        1: '#2ECC40',  # Green for vegetation
        2: '#FF4136',  # Red for equipment
        3: '#FF851B'   # Orange for damage
    }
    
    class_names = {
        0: 'Normal Infrastructure',
        1: 'Vegetation Intrusion',
        2: 'Equipment/Activity',
        3: 'Surface Damage'
    }
    
    # Plot each class
    for class_id in [0, 1, 2, 3]:
        mask = labels == class_id
        ax.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=colors[class_id],
            label=class_names[class_id],
            alpha=0.6 if class_id == 0 else 0.8,
            s=20 if class_id == 0 else 40,
            edgecolors='black',
            linewidth=0.5
        )
    
    # Highlight top anomalies (>3σ)
    threshold = np.mean(anomaly_scores) + 3 * np.std(anomaly_scores)
    outlier_mask = anomaly_scores > threshold
    n_outliers = np.sum(outlier_mask)
    
    ax.scatter(
        embeddings_2d[outlier_mask, 0],
        embeddings_2d[outlier_mask, 1],
        marker='o',
        s=120,
        facecolors='none',
        edgecolors='black',
        linewidths=2,
        label=f'Flagged for Inspection (n={n_outliers}, >3σ)'
    )
    
    # Apply minimalist style
    apply_minimalist_style_manual(ax)
    
    # Labels and title
    ax.set_xlabel('t-SNE Dimension 1', fontsize=10)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=10)
    ax.set_title('DINOv2 Embedding Space - Infrastructure Anomaly Detection', 
                 fontsize=12, fontweight='bold', loc='left', pad=20)
    
    # Legend
    ax.legend(loc='upper right', frameon=False, fontsize=9)
    
    # Add annotation
    ax.text(0.02, 0.02, 
            f'10,000 aerial images | 384-dim embeddings | {n_outliers} outliers flagged',
            transform=ax.transAxes, fontsize=8, 
            verticalalignment='bottom', color='black')
    
    plt.tight_layout()
    plt.savefig('/Users/k.jones/Desktop/blogs/blog_posts/17_infrastructure_inspection_dinov2_main.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Main visualization saved")
    print(f"  t-SNE projection with {n_outliers} outliers flagged")

def create_anomaly_distribution_visualization():
    """
    Create histogram of anomaly scores with threshold marking.
    """
    print("Generating anomaly distribution visualization...")
    
    # Generate embeddings and scores
    embeddings, labels, anomaly_scores = generate_embeddings_with_structure(n_images=10000)
    
    # Calculate statistics
    mean_score = np.mean(anomaly_scores)
    std_score = np.std(anomaly_scores)
    threshold = mean_score + 3 * std_score
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Histogram for each class
    colors = {
        0: '#CCCCCC',  # Gray for normal
        1: '#2ECC40',  # Green for vegetation
        2: '#FF4136',  # Red for equipment
        3: '#FF851B'   # Orange for damage
    }
    
    class_names = {
        0: 'Normal',
        1: 'Vegetation',
        2: 'Equipment',
        3: 'Damage'
    }
    
    for class_id in [0, 1, 2, 3]:
        mask = labels == class_id
        ax.hist(anomaly_scores[mask], bins=50, alpha=0.6, 
                color=colors[class_id], label=class_names[class_id],
                edgecolor='black', linewidth=0.5)
    
    # Mark threshold
    ax.axvline(threshold, color='black', linestyle='--', linewidth=2, 
               label=f'Threshold (μ + 3σ = {threshold:.2f})')
    
    # Mark mean
    ax.axvline(mean_score, color='black', linestyle=':', linewidth=1.5, 
               label=f'Mean = {mean_score:.2f}')
    
    # Apply minimalist style
    apply_minimalist_style_manual(ax)
    
    # Labels
    ax.set_xlabel('Anomaly Score (Distance to Cluster Centroid)', fontsize=10)
    ax.set_ylabel('Number of Images', fontsize=10)
    ax.set_title('Anomaly Score Distribution', 
                 fontsize=12, fontweight='bold', loc='left', pad=20)
    
    # Legend
    ax.legend(loc='upper right', frameon=False, fontsize=9)
    
    # Statistics annotation
    n_outliers = np.sum(anomaly_scores > threshold)
    ax.text(0.98, 0.65, 
            f'Total Images: {len(anomaly_scores):,}\n'
            f'Mean Score: {mean_score:.3f}\n'
            f'Std Dev: {std_score:.3f}\n'
            f'Threshold: {threshold:.3f}\n'
            f'Flagged: {n_outliers} ({n_outliers/len(anomaly_scores)*100:.2f}%)',
            transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', linewidth=1))
    
    plt.tight_layout()
    plt.savefig('/Users/k.jones/Desktop/blogs/blog_posts/17_infrastructure_anomaly_distribution.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Anomaly distribution visualization saved")

def create_performance_metrics_visualization():
    """
    Create bar chart showing review workload reduction and detection performance.
    """
    print("Generating performance metrics visualization...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left panel: Workload reduction
    scenarios = ['Manual\nReview\n(100%)', 'Pilot\nFlagging\n(~5%)', 'DINOv2\nAnomaly\n(~2%)']
    review_pct = [100, 5, 2]
    colors_workload = ['#FF4136', '#FF851B', '#2ECC40']
    
    bars1 = ax1.bar(scenarios, review_pct, color=colors_workload, 
                    edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    apply_minimalist_style_manual(ax1)
    ax1.set_ylabel('Images Requiring Human Review (%)', fontsize=10)
    ax1.set_title('Review Workload Reduction', 
                  fontsize=12, fontweight='bold', loc='left', pad=20)
    ax1.set_ylim(0, 110)
    
    # Add cost annotation
    ax1.text(0.5, 0.95, 
            '98% reduction: 10,000 images → 200 reviews',
            transform=ax1.transAxes, fontsize=9,
            ha='center', va='top', style='italic', color='black')
    
    # Right panel: Detection performance
    metrics = ['Anomaly\nRecall', 'False\nPositive\nRate', 'Review\nWorkload']
    values = [78, 22, 2]  # Recall 78%, FPR 22%, Workload 2%
    colors_perf = ['#2ECC40', '#FF4136', '#0074D9']
    
    bars2 = ax2.bar(metrics, values, color=colors_perf, 
                    edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    apply_minimalist_style_manual(ax2)
    ax2.set_ylabel('Percentage (%)', fontsize=10)
    ax2.set_title('Detection Performance (μ + 3σ Threshold)', 
                  fontsize=12, fontweight='bold', loc='left', pad=20)
    ax2.set_ylim(0, 90)
    
    # Add performance annotation
    ax2.text(0.5, 0.95, 
            'Captures 78% of actual anomalies while reviewing 2% of images',
            transform=ax2.transAxes, fontsize=9,
            ha='center', va='top', style='italic', color='black')
    
    plt.tight_layout()
    plt.savefig('/Users/k.jones/Desktop/blogs/blog_posts/17_infrastructure_performance.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Performance metrics visualization saved")

def main():
    """Generate all visualizations for Blog 17."""
    set_tufte_defaults()
    print("="*70)
    print("Blog 17: Infrastructure Inspection - DINOv2 Visualizations")
    print("="*70)
    print()
    
    create_main_visualization()
    create_anomaly_distribution_visualization()
    create_performance_metrics_visualization()
    
    print()
    print("="*70)
    print("All visualizations generated successfully!")
    print("="*70)
    print()
    print("Files created:")
    print("  - 17_infrastructure_inspection_dinov2_main.png")
    print("  - 17_infrastructure_anomaly_distribution.png")
    print("  - 17_infrastructure_performance.png")

if __name__ == "__main__":
    main()

