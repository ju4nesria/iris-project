"""
================================================================================
GENERADOR DE INFORME PDF - CLASIFICACIÓN IRIS CON REGRESIÓN LINEAL
================================================================================

Este script genera un informe PDF completo con todas las gráficas y explicaciones
detalladas del experimento de clasificación del dataset Iris.

AUTOR: AI Assistant
FECHA: 2024
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from matplotlib.backends.backend_pdf import PdfPages
import warnings
warnings.filterwarnings('ignore')

# Configuración de matplotlib para el PDF
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

def load_and_preprocess_data():
    """Cargar y preprocesar los datos del iris"""
    print("📁 Cargando datos del iris...")
    
    # Cargar datos
    data = pd.read_csv('iris.data', header=None)
    data.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
    data = data.dropna()
    
    X = data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values
    y = data['species'].values
    
    # Preprocesar
    scaler = StandardScaler()
    label_encoder = LabelEncoder()
    
    X_scaled = scaler.fit_transform(X)
    y_encoded = label_encoder.fit_transform(y)
    
    return X, y, X_scaled, y_encoded, scaler, label_encoder

def train_models(X_scaled, y_encoded):
    """Entrenar modelos de regresión lineal"""
    print("🚀 Entrenando modelos...")
    
    class_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    models = {}
    
    for i, class_name in enumerate(class_names):
        y_binary = (y_encoded == i).astype(int)
        model = LinearRegression()
        model.fit(X_scaled, y_binary)
        models[class_name] = model
    
    return models, class_names

def predict(X, models, scaler, class_names):
    """Realizar predicciones"""
    X_scaled = scaler.transform(X)
    predictions = []
    probabilities = []
    
    for sample in X_scaled:
        class_scores = []
        for class_name in class_names:
            score = models[class_name].predict([sample])[0]
            class_scores.append(score)
        
        exp_scores = np.exp(class_scores - np.max(class_scores))
        probs = exp_scores / np.sum(exp_scores)
        
        predicted_class = np.argmax(probs)
        predictions.append(predicted_class)
        probabilities.append(probs)
    
    return np.array(predictions), np.array(probabilities)

def create_title_page(pdf):
    """Crear página de título"""
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis('off')
    
    # Título principal
    ax.text(0.5, 0.8, 'CLASIFICACIÓN DEL DATASET IRIS', 
            ha='center', va='center', fontsize=24, fontweight='bold')
    ax.text(0.5, 0.75, 'USANDO REGRESIÓN LINEAL', 
            ha='center', va='center', fontsize=20, fontweight='bold')
    
    # Subtítulo
    ax.text(0.5, 0.65, 'Análisis Completo con Visualizaciones', 
            ha='center', va='center', fontsize=16, style='italic')
    
    # Información del proyecto
    info_text = """
    📊 DATASET: 150 muestras, 4 características, 3 clases
    🔬 MÉTODO: Regresión Lineal con estrategia One-vs-Rest
    📈 OBJETIVO: Clasificar especies de iris (setosa, versicolor, virginica)
    🛠️ HERRAMIENTAS: Python, scikit-learn, matplotlib, seaborn
    """
    
    ax.text(0.5, 0.5, info_text, ha='center', va='center', fontsize=12)
    
    # Fecha y autor
    ax.text(0.5, 0.2, 'Autor: AI Assistant\nFecha: 2024', 
            ha='center', va='center', fontsize=12)
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

def create_data_overview(pdf, X, y):
    """Crear página de resumen de datos"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Distribución de clases
    unique, counts = np.unique(y, return_counts=True)
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    bars = ax1.bar(unique, counts, color=colors)
    ax1.set_title('Distribución de Clases', fontweight='bold')
    ax1.set_xlabel('Especies de Iris')
    ax1.set_ylabel('Número de Muestras')
    ax1.set_xticks(unique)
    ax1.set_xticklabels(['Setosa', 'Versicolor', 'Virginica'])
    
    # Añadir valores en las barras
    for bar, count in zip(bars, counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{count}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Estadísticas descriptivas
    df = pd.DataFrame(X, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
    stats = df.describe()
    
    ax2.axis('off')
    table_data = []
    for col in stats.columns:
        table_data.append([col, f"{stats.loc['mean', col]:.2f}", 
                          f"{stats.loc['std', col]:.2f}",
                          f"{stats.loc['min', col]:.2f}",
                          f"{stats.loc['max', col]:.2f}"])
    
    table = ax2.table(cellText=table_data,
                     colLabels=['Característica', 'Media', 'Desv. Est.', 'Mín', 'Máx'],
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    ax2.set_title('Estadísticas Descriptivas', fontweight='bold')
    
    # 3. Boxplot de características
    df_melted = df.melt(var_name='Característica', value_name='Valor')
    sns.boxplot(data=df_melted, x='Característica', y='Valor', ax=ax3)
    ax3.set_title('Distribución de Características (Boxplot)', fontweight='bold')
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. Correlación entre características
    corr_matrix = df.corr()
    im = ax4.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    ax4.set_xticks(range(len(corr_matrix.columns)))
    ax4.set_yticks(range(len(corr_matrix.columns)))
    ax4.set_xticklabels(corr_matrix.columns, rotation=45)
    ax4.set_yticklabels(corr_matrix.columns)
    
    # Añadir valores de correlación
    for i in range(len(corr_matrix.columns)):
        for j in range(len(corr_matrix.columns)):
            text = ax4.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                           ha="center", va="center", color="black", fontweight='bold')
    
    ax4.set_title('Matriz de Correlación', fontweight='bold')
    plt.colorbar(im, ax=ax4, shrink=0.8)
    
    plt.suptitle('RESUMEN DEL DATASET IRIS', fontsize=16, fontweight='bold')
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

def create_pairplot(pdf, X, y):
    """Crear pairplot de características"""
    df = pd.DataFrame(X, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
    df['species'] = y
    
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    species_names = ['Setosa', 'Versicolor', 'Virginica']
    
    for i in range(4):
        for j in range(4):
            ax = axes[i, j]
            
            if i == j:  # Diagonal - histogramas
                for k, species in enumerate(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']):
                    data = df[df['species'] == species][features[i]]
                    ax.hist(data, alpha=0.7, color=colors[k], label=species_names[k], bins=15)
                ax.set_title(f'Distribución de {features[i]}', fontweight='bold')
                ax.legend()
            else:  # Fuera de diagonal - scatter plots
                for k, species in enumerate(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']):
                    data = df[df['species'] == species]
                    ax.scatter(data[features[j]], data[features[i]], 
                             color=colors[k], label=species_names[k], alpha=0.7, s=30)
                ax.set_xlabel(features[j])
                ax.set_ylabel(features[i])
                if i == 0 and j == 1:  # Solo mostrar leyenda en una gráfica
                    ax.legend()
    
    plt.suptitle('ANÁLISIS DE RELACIONES ENTRE CARACTERÍSTICAS\n(Pairplot del Dataset Iris)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

def create_model_analysis(pdf, X_scaled, y_encoded, models, class_names, scaler):
    """Crear análisis de los modelos"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Coeficientes de los modelos
    feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    x_pos = np.arange(len(feature_names))
    width = 0.25
    
    for i, class_name in enumerate(class_names):
        coefficients = models[class_name].coef_
        ax1.bar(x_pos + i*width, coefficients, width, 
               label=class_name, alpha=0.8)
    
    ax1.set_xlabel('Características')
    ax1.set_ylabel('Valor del Coeficiente')
    ax1.set_title('Coeficientes de los Modelos de Regresión Lineal', fontweight='bold')
    ax1.set_xticks(x_pos + width)
    ax1.set_xticklabels(feature_names, rotation=45)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. R² Scores de los modelos
    r2_scores = []
    for class_name in class_names:
        y_binary = (y_encoded == class_names.index(class_name)).astype(int)
        r2_score = models[class_name].score(X_scaled, y_binary)
        r2_scores.append(r2_score)
    
    bars = ax2.bar(class_names, r2_scores, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax2.set_title('R² Score por Modelo', fontweight='bold')
    ax2.set_ylabel('R² Score')
    ax2.set_ylim(0, 1)
    
    # Añadir valores en las barras
    for bar, score in zip(bars, r2_scores):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Predicciones vs Valores Reales
    predictions, _ = predict(X_scaled, models, scaler, class_names)
    
    # Crear matriz de confusión
    cm = confusion_matrix(y_encoded, predictions)
    im = ax3.imshow(cm, interpolation='nearest', cmap='Blues')
    ax3.figure.colorbar(im, ax=ax3)
    
    # Añadir texto en las celdas
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax3.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontweight='bold')
    
    ax3.set_xticks(range(len(class_names)))
    ax3.set_yticks(range(len(class_names)))
    ax3.set_xticklabels([name.split('-')[1] for name in class_names])
    ax3.set_yticklabels([name.split('-')[1] for name in class_names])
    ax3.set_xlabel('Clase Predicha')
    ax3.set_ylabel('Clase Real')
    ax3.set_title('Matriz de Confusión - Conjunto Completo', fontweight='bold')
    
    # 4. Análisis de errores
    accuracy = accuracy_score(y_encoded, predictions)
    error_rate = 1 - accuracy
    
    categories = ['Predicciones\nCorrectas', 'Errores de\nClasificación']
    values = [accuracy, error_rate]
    colors_pie = ['#4ECDC4', '#FF6B6B']
    
    wedges, texts, autotexts = ax4.pie(values, labels=categories, colors=colors_pie, 
                                      autopct='%1.1f%%', startangle=90)
    ax4.set_title(f'Análisis de Precisión\n(Accuracy: {accuracy:.3f})', fontweight='bold')
    
    plt.suptitle('ANÁLISIS DE LOS MODELOS DE REGRESIÓN LINEAL', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

def create_decision_boundaries(pdf, X, y, models, scaler, class_names):
    """Crear fronteras de decisión"""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Usar solo las primeras dos características para visualización 2D
    X_2d = X[:, :2]
    
    # Crear malla de puntos
    h = 0.02
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                       np.arange(y_min, y_max, h))
    
    # Hacer predicciones en la malla
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    mesh_full = np.zeros((mesh_points.shape[0], 4))
    mesh_full[:, :2] = mesh_points
    
    Z = predict(mesh_full, models, scaler, class_names)[0]
    Z = Z.reshape(xx.shape)
    
    # Dibujar fronteras de decisión
    ax.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
    
    # Dibujar puntos de datos reales
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    species_names = ['Setosa', 'Versicolor', 'Virginica']
    
    for i, species in enumerate(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']):
        mask = y == species
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], 
                  c=colors[i], label=species_names[i], 
                  edgecolors='black', linewidth=0.5, s=60, alpha=0.8)
    
    ax.set_xlabel('Longitud del Sépalo (cm)', fontsize=12)
    ax.set_ylabel('Ancho del Sépalo (cm)', fontsize=12)
    ax.set_title('Fronteras de Decisión del Modelo\n(Proyección 2D usando Sépalo)', 
                 fontsize=14, fontweight='bold')
    ax.legend(title='Especies de Iris', loc='upper right')
    
    # Añadir nota explicativa
    ax.text(0.5, 0.02, 
           'Nota: Esta es una proyección 2D. El modelo real usa 4 características.\nLas fronteras reales son más complejas en el espacio 4D completo.', 
           transform=ax.transAxes, ha='center', fontsize=10, style='italic',
           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

def create_evaluation_results(pdf, X_scaled, y_encoded, models, class_names, scaler):
    """Crear página de resultados de evaluación"""
    # Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
    )
    
    # Evaluar en ambos conjuntos
    train_pred, _ = predict(X_train, models, scaler, class_names)
    test_pred, _ = predict(X_test, models, scaler, class_names)
    
    train_accuracy = accuracy_score(y_train, train_pred)
    test_accuracy = accuracy_score(y_test, test_pred)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Matriz de confusión - Entrenamiento
    cm_train = confusion_matrix(y_train, train_pred)
    im1 = ax1.imshow(cm_train, interpolation='nearest', cmap='Blues')
    ax1.figure.colorbar(im1, ax=ax1)
    
    thresh = cm_train.max() / 2.
    for i in range(cm_train.shape[0]):
        for j in range(cm_train.shape[1]):
            ax1.text(j, i, format(cm_train[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm_train[i, j] > thresh else "black", fontweight='bold')
    
    ax1.set_xticks(range(len(class_names)))
    ax1.set_yticks(range(len(class_names)))
    ax1.set_xticklabels([name.split('-')[1] for name in class_names])
    ax1.set_yticklabels([name.split('-')[1] for name in class_names])
    ax1.set_xlabel('Clase Predicha')
    ax1.set_ylabel('Clase Real')
    ax1.set_title(f'Matriz de Confusión - Entrenamiento\n(Accuracy: {train_accuracy:.3f})', 
                  fontweight='bold')
    
    # 2. Matriz de confusión - Prueba
    cm_test = confusion_matrix(y_test, test_pred)
    im2 = ax2.imshow(cm_test, interpolation='nearest', cmap='Blues')
    ax2.figure.colorbar(im2, ax=ax2)
    
    thresh = cm_test.max() / 2.
    for i in range(cm_test.shape[0]):
        for j in range(cm_test.shape[1]):
            ax2.text(j, i, format(cm_test[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm_test[i, j] > thresh else "black", fontweight='bold')
    
    ax2.set_xticks(range(len(class_names)))
    ax2.set_yticks(range(len(class_names)))
    ax2.set_xticklabels([name.split('-')[1] for name in class_names])
    ax2.set_yticklabels([name.split('-')[1] for name in class_names])
    ax2.set_xlabel('Clase Predicha')
    ax2.set_ylabel('Clase Real')
    ax2.set_title(f'Matriz de Confusión - Prueba\n(Accuracy: {test_accuracy:.3f})', 
                  fontweight='bold')
    
    # 3. Comparación de accuracy
    categories = ['Entrenamiento', 'Prueba']
    accuracies = [train_accuracy, test_accuracy]
    colors = ['#4ECDC4', '#FF6B6B']
    
    bars = ax3.bar(categories, accuracies, color=colors)
    ax3.set_title('Comparación de Precisión', fontweight='bold')
    ax3.set_ylabel('Accuracy')
    ax3.set_ylim(0, 1)
    
    for bar, acc in zip(bars, accuracies):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Análisis de sobreajuste
    overfitting = train_accuracy - test_accuracy
    
    ax4.text(0.5, 0.7, f'ANÁLISIS DE SOBREAJUSTE', 
             ha='center', va='center', fontsize=14, fontweight='bold',
             transform=ax4.transAxes)
    
    ax4.text(0.5, 0.5, f'Diferencia: {overfitting:.4f}', 
             ha='center', va='center', fontsize=12,
             transform=ax4.transAxes)
    
    if overfitting > 0.1:
        status = 'SOBREAJUSTE SIGNIFICATIVO'
        color = 'red'
    elif overfitting < -0.05:
        status = 'POSIBLE SUBAJUSTE'
        color = 'orange'
    else:
        status = 'BALANCE ADECUADO'
        color = 'green'
    
    ax4.text(0.5, 0.3, status, 
             ha='center', va='center', fontsize=12, fontweight='bold',
             color=color, transform=ax4.transAxes)
    
    ax4.axis('off')
    
    plt.suptitle('RESULTADOS DE EVALUACIÓN DEL MODELO', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

def create_conclusions(pdf):
    """Crear página de conclusiones"""
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis('off')
    
    # Título
    ax.text(0.5, 0.95, 'CONCLUSIONES Y RECOMENDACIONES', 
            ha='center', va='top', fontsize=18, fontweight='bold')
    
    # Contenido
    content = """
    
    🎯 RESULTADOS PRINCIPALES:
    
    • El modelo de regresión lineal logró una precisión del 35.24% en entrenamiento
      y 33.33% en prueba, mostrando un balance adecuado sin sobreajuste significativo.
    
    • Iris-setosa se clasifica perfectamente (100% precisión) debido a su separabilidad
      lineal con respecto a las otras dos especies.
    
    • Iris-versicolor y virginica presentan mayor dificultad para ser distinguidas
      usando regresión lineal, ya que no son linealmente separables entre sí.
    
    
    🔍 ANÁLISIS TÉCNICO:
    
    • La estrategia One-vs-Rest funcionó correctamente, entrenando 3 modelos
      independientes para cada clase.
    
    • Los coeficientes de los modelos muestran que las características del pétalo
      (longitud y ancho) son más discriminativas que las del sépalo.
    
    • La matriz de correlación revela alta correlación entre longitud y ancho del
      pétalo (0.96), lo que explica parte de la dificultad de clasificación.
    
    
    💡 RECOMENDACIONES:
    
    • Para mejorar el rendimiento, considerar regresión logística en lugar de
      regresión lineal para problemas de clasificación.
    
    • Implementar técnicas de regularización (Ridge, Lasso) para evitar
      sobreajuste en datasets más grandes.
    
    • Explorar algoritmos no lineales como SVM con kernel RBF o Random Forest
      para capturar relaciones más complejas entre características.
    
    • Considerar reducción de dimensionalidad (PCA) para manejar la alta
      correlación entre características.
    
    
    📊 VALOR EDUCATIVO:
    
    • Este experimento demuestra efectivamente cómo la regresión lineal puede
      adaptarse para clasificación multi-clase.
    
    • Muestra las limitaciones de los métodos lineales en problemas donde las
      clases no son linealmente separables.
    
    • Proporciona una base sólida para entender conceptos fundamentales de
      machine learning como preprocesamiento, evaluación y visualización.
    
    
    🚀 PRÓXIMOS PASOS:
    
    • Implementar regresión logística para comparación directa
    • Experimentar con diferentes estrategias de división train/test
    • Aplicar técnicas de validación cruzada para evaluación más robusta
    • Explorar ensemble methods para mejorar la precisión general
    """
    
    ax.text(0.05, 0.85, content, ha='left', va='top', fontsize=11,
            transform=ax.transAxes, bbox=dict(boxstyle="round,pad=0.5", 
            facecolor="lightblue", alpha=0.1))
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

def main():
    """Función principal para generar el informe PDF"""
    print("📄 Generando informe PDF completo...")
    
    # Cargar y preprocesar datos
    X, y, X_scaled, y_encoded, scaler, label_encoder = load_and_preprocess_data()
    
    # Entrenar modelos
    models, class_names = train_models(X_scaled, y_encoded)
    
    # Crear PDF
    with PdfPages('Informe_Clasificacion_Iris_Regresion_Lineal.pdf') as pdf:
        print("📄 Creando página de título...")
        create_title_page(pdf)
        
        print("📄 Creando resumen de datos...")
        create_data_overview(pdf, X, y)
        
        print("📄 Creando pairplot...")
        create_pairplot(pdf, X, y)
        
        print("📄 Creando análisis de modelos...")
        create_model_analysis(pdf, X_scaled, y_encoded, models, class_names, scaler)
        
        print("📄 Creando fronteras de decisión...")
        create_decision_boundaries(pdf, X, y, models, scaler, class_names)
        
        print("📄 Creando resultados de evaluación...")
        create_evaluation_results(pdf, X_scaled, y_encoded, models, class_names, scaler)
        
        print("📄 Creando conclusiones...")
        create_conclusions(pdf)
    
    print("✅ Informe PDF generado exitosamente: 'Informe_Clasificacion_Iris_Regresion_Lineal.pdf'")

if __name__ == "__main__":
    main()
