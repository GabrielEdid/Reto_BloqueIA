"""
Visualización de Resultados de docTR en CORD-v2
Genera tablas y gráficas para la presentación
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Configuración de estilo
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_results(csv_path):
    """Carga los resultados de docTR desde CSV"""
    df = pd.read_csv(csv_path)
    return df

def print_general_metrics(metrics_path):
    """Imprime tabla de métricas generales"""
    metrics = pd.read_csv(metrics_path)
    
    print("\n" + "="*70)
    print("          docTR Resultados - Configuración Default")
    print("              (DB-ResNet50 + CRNN-VGG16-BN)")
    print("="*70)
    print("Dataset: CORD-v2 (100 recibos de prueba)")
    print("-"*70)
    
    for _, row in metrics.iterrows():
        metric_name = row['metric'].replace('_', ' ').title()
        value = row['value']
        
        if 'accuracy' in row['metric']:
            print(f"{metric_name:.<50} {float(value)*100:>6.2f}%")
        else:
            print(f"{metric_name:.<50} {int(float(value)):>6}")
    
    print("="*70)

def print_top_predictions(df, top_n=10):
    """Imprime tabla con las mejores predicciones"""
    # Ordenar por accuracy descendente
    df_sorted = df.sort_values('accuracy', ascending=False)
    
    print("\n" + "="*100)
    print(f"Top {top_n} Mejores Predicciones")
    print("="*100)
    print(f"{'ID':<5} {'Ground Truth (primeros 50 chars)':<55} {'Precisión':>12}")
    print("-"*100)
    
    for idx, row in df_sorted.head(top_n).iterrows():
        sample_id = row['sample_id']
        gt_preview = row['ground_truth'][:50] + "..." if len(row['ground_truth']) > 50 else row['ground_truth']
        accuracy = float(row['accuracy']) * 100
        print(f"{sample_id:<5} {gt_preview:<55} {accuracy:>11.2f}%")
    
    print("="*100)

def print_accuracy_distribution(df):
    """Imprime tabla de distribución de precisión por rangos"""
    df['accuracy_pct'] = df['accuracy'] * 100
    
    ranges = [
        (95, 100, "95-100%"),
        (90, 95, "90-95%"),
        (85, 90, "85-90%"),
        (80, 85, "80-85%"),
        (0, 80, "< 80%")
    ]
    
    print("\n" + "="*60)
    print("Distribución de Precisión por Rangos")
    print("="*60)
    print(f"{'Rango Precisión':<20} {'# Muestras':>12} {'Porcentaje':>12}")
    print("-"*60)
    
    total = len(df)
    for min_val, max_val, label in ranges:
        if min_val == 0:
            count = len(df[df['accuracy_pct'] < max_val])
        else:
            count = len(df[(df['accuracy_pct'] >= min_val) & (df['accuracy_pct'] < max_val)])
        percentage = (count / total) * 100
        print(f"{label:<20} {count:>12} {percentage:>11.1f}%")
    
    print("-"*60)
    print(f"{'TOTAL':<20} {total:>12} {'100.0%':>12}")
    print("="*60)

def print_comparison_table():
    """Imprime tabla comparativa de todos los modelos"""
    data = {
        'Modelo': [
            'PyTesseract',
            'EasyOCR',
            'docTR Default ⭐',
            'TrOCR S1 (Frozen)',
            'TrOCR S2 (Partial)',
            'TrOCR S3 (Full)',
            'Donut S1 (Frozen)',
            'Donut S2 (Partial)',
            'Donut S3 (Full)'
        ],
        'Precisión': [17, 17, 90.92, 92, 95, 97, 94, 96, 97],
        'Fine-tuning': ['❌', '❌', '❌', '✅', '✅', '✅', '✅', '✅', '✅'],
        'Salida': [
            'Texto plano', 'Texto plano', 'Texto plano',
            'Texto plano', 'Texto plano', 'Texto plano',
            'JSON', 'JSON', 'JSON'
        ],
        'Velocidad': [
            'Rápido (CPU)', 'Medio (GPU)', 'Rápido (GPU)',
            'Medio', 'Medio', 'Medio',
            'Medio', 'Medio', 'Medio'
        ]
    }
    
    df = pd.DataFrame(data)
    
    print("\n" + "="*100)
    print("Comparación de Modelos OCR en CORD-v2")
    print("="*100)
    print(f"{'Modelo':<25} {'Precisión':>10} {'Fine-tuning':>13} {'Salida':<15} {'Velocidad':<15}")
    print("-"*100)
    
    for _, row in df.iterrows():
        print(f"{row['Modelo']:<25} {row['Precisión']:>9.2f}% {row['Fine-tuning']:>13} {row['Salida']:<15} {row['Velocidad']:<15}")
    
    print("="*100)

def plot_accuracy_distribution(df, save_path=None):
    """Genera gráfica de distribución de precisión"""
    df['accuracy_pct'] = df['accuracy'] * 100
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Histograma
    ax1.hist(df['accuracy_pct'], bins=20, edgecolor='black', alpha=0.7, color='skyblue')
    ax1.axvline(df['accuracy_pct'].mean(), color='red', linestyle='--', linewidth=2, label=f'Promedio: {df["accuracy_pct"].mean():.2f}%')
    ax1.set_xlabel('Precisión (%)', fontsize=12)
    ax1.set_ylabel('Número de Muestras', fontsize=12)
    ax1.set_title('Distribución de Precisión - docTR en CORD-v2', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Box plot
    ax2.boxplot(df['accuracy_pct'], vert=True, patch_artist=True,
                boxprops=dict(facecolor='lightblue', alpha=0.7),
                medianprops=dict(color='red', linewidth=2))
    ax2.set_ylabel('Precisión (%)', fontsize=12)
    ax2.set_title('Distribución Estadística de Precisión', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_xticklabels(['docTR Default'])
    
    # Añadir estadísticas
    stats_text = f"Media: {df['accuracy_pct'].mean():.2f}%\n"
    stats_text += f"Mediana: {df['accuracy_pct'].median():.2f}%\n"
    stats_text += f"Desv. Est.: {df['accuracy_pct'].std():.2f}%\n"
    stats_text += f"Mín: {df['accuracy_pct'].min():.2f}%\n"
    stats_text += f"Máx: {df['accuracy_pct'].max():.2f}%"
    
    ax2.text(0.02, 0.98, stats_text, transform=fig.transFigure,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Gráfica guardada en: {save_path}")
    
    plt.show()

def plot_model_comparison(save_path=None):
    """Genera gráfica comparativa de modelos"""
    models = [
        'PyTesseract',
        'EasyOCR',
        'docTR\nDefault ⭐',
        'TrOCR\nS1',
        'TrOCR\nS2',
        'TrOCR\nS3',
        'Donut\nS1',
        'Donut\nS2',
        'Donut\nS3'
    ]
    
    accuracies = [17, 17, 90.92, 92, 95, 97, 94, 96, 97]
    
    colors = ['#ff9999', '#ff9999', '#66b3ff', '#99ff99', '#99ff99', '#99ff99', '#ffcc99', '#ffcc99', '#ffcc99']
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    bars = ax.barh(models, accuracies, color=colors, edgecolor='black', alpha=0.8)
    
    # Añadir valores en las barras
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        width = bar.get_width()
        ax.text(width + 1, bar.get_y() + bar.get_height()/2,
                f'{acc:.2f}%',
                ha='left', va='center', fontweight='bold', fontsize=11)
    
    ax.set_xlabel('Precisión (%)', fontsize=13, fontweight='bold')
    ax.set_title('Comparación de Precisión en CORD-v2 (100 recibos)', fontsize=15, fontweight='bold')
    ax.set_xlim(0, 105)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Línea de referencia
    ax.axvline(90, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label='90% (Umbral bueno)')
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Gráfica guardada en: {save_path}")
    
    plt.show()

def plot_accuracy_ranges(df, save_path=None):
    """Gráfica de pie con rangos de precisión"""
    df['accuracy_pct'] = df['accuracy'] * 100
    
    ranges = [
        (95, 100, "95-100% (Excelente)"),
        (90, 95, "90-95% (Muy bueno)"),
        (85, 90, "85-90% (Bueno)"),
        (80, 85, "80-85% (Regular)"),
        (0, 80, "< 80% (Pobre)")
    ]
    
    counts = []
    labels = []
    
    for min_val, max_val, label in ranges:
        if min_val == 0:
            count = len(df[df['accuracy_pct'] < max_val])
        else:
            count = len(df[(df['accuracy_pct'] >= min_val) & (df['accuracy_pct'] < max_val)])
        counts.append(count)
        labels.append(f"{label}\n{count} recibos")
    
    colors_pie = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c', '#95a5a6']
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    wedges, texts, autotexts = ax.pie(
        counts, 
        labels=labels, 
        autopct='%1.1f%%',
        colors=colors_pie,
        startangle=90,
        textprops={'fontsize': 11, 'weight': 'bold'}
    )
    
    ax.set_title('Distribución de Precisión por Rangos\ndocTR en CORD-v2', 
                 fontsize=15, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Gráfica guardada en: {save_path}")
    
    plt.show()

def generate_all_visualizations():
    """Genera todas las tablas y gráficas"""
    # Paths
    results_csv = "./results_doctr_20251203_194256/test_predictions.csv"
    metrics_csv = "./results_doctr_20251203_194256/metrics.csv"
    output_dir = Path("./visualizations")
    output_dir.mkdir(exist_ok=True)
    
    # Cargar datos
    print("\n📊 Cargando resultados de docTR...")
    df_results = load_results(results_csv)
    
    # Tablas
    print("\n" + "🔢 GENERANDO TABLAS...")
    print_general_metrics(metrics_csv)
    print_top_predictions(df_results, top_n=10)
    print_accuracy_distribution(df_results)
    print_comparison_table()
    
    # Gráficas
    print("\n" + "📈 GENERANDO GRÁFICAS...")
    
    print("\n[1/3] Distribución de precisión...")
    plot_accuracy_distribution(
        df_results, 
        save_path=output_dir / "accuracy_distribution.png"
    )
    
    print("\n[2/3] Comparación de modelos...")
    plot_model_comparison(
        save_path=output_dir / "model_comparison.png"
    )
    
    print("\n[3/3] Rangos de precisión...")
    plot_accuracy_ranges(
        df_results,
        save_path=output_dir / "accuracy_ranges.png"
    )
    
    print("\n" + "="*70)
    print("✅ TODAS LAS VISUALIZACIONES GENERADAS EXITOSAMENTE")
    print("="*70)
    print(f"\nArchivos guardados en: {output_dir.absolute()}/")
    print("  - accuracy_distribution.png")
    print("  - model_comparison.png")
    print("  - accuracy_ranges.png")
    print("="*70)

if __name__ == "__main__":
    generate_all_visualizations()
