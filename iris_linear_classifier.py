"""
================================================================================
CLASIFICACIÓN DEL DATASET IRIS USANDO REGRESIÓN LINEAL
================================================================================

Este script implementa un clasificador completo para el dataset de Iris utilizando
únicamente regresión lineal con estrategia One-vs-Rest.

AUTOR: AI Assistant
FECHA: 2025
DESCRIPCIÓN: Clasificación multi-clase usando regresión lineal para 3 especies de iris
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Configuración de matplotlib para mejor visualización
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

class IrisLinearClassifier:
    """
    ================================================================================
    CLASIFICADOR DE REGRESIÓN LINEAL PARA EL DATASET IRIS
    ================================================================================
    
    Esta clase implementa un clasificador multi-clase usando regresión lineal
    con estrategia One-vs-Rest para las tres especies de iris:
    - Iris-setosa
    - Iris-versicolor  
    - Iris-virginica
    
    MÉTODO:
    1. Entrena 3 modelos de regresión lineal (uno por cada clase)
    2. Cada modelo predice la probabilidad de pertenecer a su clase específica
    3. La clase final se determina por la probabilidad más alta
    
    CARACTERÍSTICAS:
    - Preprocesamiento automático de datos
    - Estandarización de características
    - Evaluación completa con métricas
    - Visualizaciones detalladas
    - Documentación exhaustiva
    """
    
    def __init__(self, use_regularization=False, alpha=1.0):
        """
        ================================================================================
        INICIALIZAR EL CLASIFICADOR
        ================================================================================
        
        PARÁMETROS:
        -----------
        use_regularization : bool, default=False
            Si usar regresión Ridge (regularizada) en lugar de regresión lineal estándar
        alpha : float, default=1.0
            Parámetro de regularización para Ridge (solo si use_regularization=True)
            
        ATRIBUTOS:
        ----------
        scaler : StandardScaler
            Estandarizador para normalizar las características
        label_encoder : LabelEncoder
            Codificador para convertir etiquetas de texto a números
        models : dict
            Diccionario que almacena los modelos entrenados para cada clase
        class_names : list
            Nombres de las tres clases de iris
        feature_names : list
            Nombres de las cuatro características del dataset
        """
        print("🔧 Inicializando clasificador de regresión lineal...")
        
        # Inicializar componentes de preprocesamiento
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # Configuración del modelo
        self.use_regularization = use_regularization
        self.alpha = alpha
        
        # Nombres de clases y características
        self.class_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
        self.feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        
        # Almacenar modelos entrenados
        self.models = {}
        
        print(f"✅ Clasificador inicializado - Regularización: {use_regularization}")
    
    def load_data(self, file_path='iris.data'):
        """
        ================================================================================
        CARGAR Y PREPARAR EL DATASET IRIS
        ================================================================================
        
        Esta función carga el dataset de iris desde un archivo CSV y lo prepara
        para el entrenamiento del modelo.
        
        PARÁMETROS:
        -----------
        file_path : str, default='iris.data'
            Ruta al archivo de datos del dataset iris
            
        RETORNA:
        --------
        X : numpy.ndarray
            Matriz de características (150, 4) con las medidas de las flores
        y : numpy.ndarray
            Vector de etiquetas (150,) con las especies de iris
            
        INFORMACIÓN DEL DATASET:
        ------------------------
        - 150 muestras totales (50 por cada clase)
        - 4 características: longitud y ancho del sépalo y pétalo
        - 3 clases: setosa, versicolor, virginica
        - Sin valores faltantes
        """
        print(f"📁 Cargando datos desde: {file_path}")
        
        try:
            # Leer el archivo CSV
            data = pd.read_csv(file_path, header=None)
            
            # Asignar nombres a las columnas
            data.columns = self.feature_names + ['species']
            
            # Eliminar filas vacías si las hay
            data = data.dropna()
            
            # Separar características (X) y etiquetas (y)
            X = data[self.feature_names].values
            y = data['species'].values
            
            # Mostrar información del dataset
            print(f"✅ Dataset cargado exitosamente")
            print(f"   📊 Forma del dataset: {X.shape}")
            print(f"   🏷️  Clases encontradas: {np.unique(y)}")
            print(f"   📈 Distribución de clases: {np.bincount(self.label_encoder.fit_transform(y))}")
            
            return X, y
            
        except FileNotFoundError:
            print(f"❌ Error: No se pudo encontrar el archivo {file_path}")
            raise
        except Exception as e:
            print(f"❌ Error al cargar los datos: {str(e)}")
            raise
    
    def preprocess_data(self, X, y):
        """
        ================================================================================
        PREPROCESAR LOS DATOS PARA EL ENTRENAMIENTO
        ================================================================================
        
        Esta función prepara los datos para el entrenamiento del modelo:
        1. Estandariza las características (media=0, desviación=1)
        2. Codifica las etiquetas de texto a números
        
        PARÁMETROS:
        -----------
        X : numpy.ndarray
            Matriz de características originales
        y : numpy.ndarray
            Vector de etiquetas originales (texto)
            
        RETORNA:
        --------
        X_scaled : numpy.ndarray
            Características estandarizadas
        y_encoded : numpy.ndarray
            Etiquetas codificadas numéricamente
            
        IMPORTANCIA DE LA ESTANDARIZACIÓN:
        ----------------------------------
        Las características tienen diferentes escalas:
        - Sépalo: 4.3-7.9 cm
        - Pétalo: 1.0-6.9 cm
        La estandarización evita que características con valores más grandes
        dominen el modelo.
        """
        print("🔧 Preprocesando datos...")
        
        # Estandarizar características (media=0, desviación=1)
        X_scaled = self.scaler.fit_transform(X)
        
        # Codificar etiquetas de texto a números
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Mostrar estadísticas de preprocesamiento
        print("✅ Preprocesamiento completado")
        print(f"   📊 Características estandarizadas - Media: {X_scaled.mean():.3f}, Desv: {X_scaled.std():.3f}")
        print(f"   🏷️  Etiquetas codificadas: {np.unique(y_encoded)}")
        
        return X_scaled, y_encoded
    
    def train_models(self, X, y):
        """
        ================================================================================
        ENTRENAR MODELOS DE REGRESIÓN LINEAL
        ================================================================================
        
        Esta función entrena un modelo de regresión lineal separado para cada clase
        usando la estrategia One-vs-Rest (Uno contra Todos).
        
        ESTRATEGIA ONE-VS-REST:
        ----------------------
        Para cada clase i:
        1. Crear etiquetas binarias: 1 si pertenece a la clase i, 0 si no
        2. Entrenar un modelo de regresión lineal para predecir estas etiquetas
        3. El modelo aprende a distinguir la clase i del resto
        
        PARÁMETROS:
        -----------
        X : numpy.ndarray
            Características estandarizadas de entrenamiento
        y : numpy.ndarray
            Etiquetas codificadas de entrenamiento
            
        MODELOS ENTRENADOS:
        -------------------
        - Modelo 0: Distingue Iris-setosa del resto
        - Modelo 1: Distingue Iris-versicolor del resto  
        - Modelo 2: Distingue Iris-virginica del resto
        """
        print("🚀 Entrenando modelos de regresión lineal...")
        print("   📋 Estrategia: One-vs-Rest (Uno contra Todos)")
        
        for i, class_name in enumerate(self.class_names):
            print(f"\n   🔄 Entrenando modelo para {class_name}...")
            
            # Crear etiquetas binarias para esta clase
            y_binary = (y == i).astype(int)
            
            # Seleccionar tipo de modelo
            if self.use_regularization:
                model = Ridge(alpha=self.alpha)
                print(f"      📐 Usando regresión Ridge (α={self.alpha})")
            else:
                model = LinearRegression()
                print(f"      📐 Usando regresión lineal estándar")
            
            # Entrenar el modelo
            model.fit(X, y_binary)
            
            # Calcular y mostrar R² score
            r2_score = model.score(X, y_binary)
            print(f"      ✅ Modelo entrenado - R² Score: {r2_score:.4f}")
            
            # Almacenar el modelo
            self.models[class_name] = model
        
        print(f"\n✅ Todos los modelos entrenados exitosamente")
        print(f"   📊 Total de modelos: {len(self.models)}")
    
    def predict(self, X):
        """
        ================================================================================
        REALIZAR PREDICCIONES CON LOS MODELOS ENTRENADOS
        ================================================================================
        
        Esta función utiliza los modelos entrenados para predecir la clase
        de nuevas muestras de iris.
        
        PROCESO DE PREDICCIÓN:
        ----------------------
        1. Estandarizar las características de entrada
        2. Obtener puntuación de cada modelo (probabilidad de pertenecer a cada clase)
        3. Aplicar softmax para convertir puntuaciones en probabilidades
        4. Seleccionar la clase con mayor probabilidad
        
        PARÁMETROS:
        -----------
        X : numpy.ndarray
            Características a predecir (pueden ser múltiples muestras)
            
        RETORNA:
        --------
        predictions : numpy.ndarray
            Clases predichas (0, 1, o 2)
        probabilities : numpy.ndarray
            Matriz de probabilidades para cada clase
        """
        # Estandarizar las características de entrada
        X_scaled = self.scaler.transform(X)
        
        predictions = []
        probabilities = []
        
        # Para cada muestra
        for sample in X_scaled:
            class_scores = []
            
            # Obtener puntuación de cada modelo
            for class_name in self.class_names:
                score = self.models[class_name].predict([sample])[0]
                class_scores.append(score)
            
            # Convertir puntuaciones a probabilidades usando softmax
            # Softmax: exp(xi) / sum(exp(xj)) para j=1..n
            exp_scores = np.exp(class_scores - np.max(class_scores))  # Estabilidad numérica
            probs = exp_scores / np.sum(exp_scores)
            
            # Predecir la clase con mayor probabilidad
            predicted_class = np.argmax(probs)
            predictions.append(predicted_class)
            probabilities.append(probs)
        
        return np.array(predictions), np.array(probabilities)
    
    def evaluate(self, X, y_true):
        """
        ================================================================================
        EVALUAR EL RENDIMIENTO DEL MODELO
        ================================================================================
        
        Esta función evalúa qué tan bien funciona el modelo usando varias métricas
        de clasificación estándar.
        
        MÉTRICAS CALCULADAS:
        --------------------
        - Accuracy: Porcentaje de predicciones correctas
        - Precision: De las predicciones positivas, cuántas son correctas
        - Recall: De los casos positivos reales, cuántos detectó el modelo
        - F1-Score: Media armónica entre precision y recall
        - Matriz de confusión: Detalle de aciertos y errores por clase
        
        PARÁMETROS:
        -----------
        X : numpy.ndarray
            Características para evaluar
        y_true : numpy.ndarray
            Etiquetas verdaderas (codificadas numéricamente)
            
        RETORNA:
        --------
        results : dict
            Diccionario con todas las métricas de evaluación
        """
        print("📊 Evaluando rendimiento del modelo...")
        
        # Realizar predicciones
        y_pred, y_probs = self.predict(X)
        
        # Calcular accuracy
        accuracy = accuracy_score(y_true, y_pred)
        
        # Convertir etiquetas numéricas a nombres para el reporte
        y_true_names = self.label_encoder.inverse_transform(y_true)
        y_pred_names = self.label_encoder.inverse_transform(y_pred)
        
        # Generar reporte de clasificación
        report = classification_report(y_true_names, y_pred_names, 
                                    target_names=self.class_names, 
                                    output_dict=True)
        
        # Calcular matriz de confusión
        cm = confusion_matrix(y_true, y_pred)
        
        # Compilar resultados
        results = {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'probabilities': y_probs
        }
        
        print(f"✅ Evaluación completada - Accuracy: {accuracy:.4f}")
        
        return results
    
    def plot_confusion_matrix(self, cm, title="Matriz de Confusión"):
        """
        ================================================================================
        VISUALIZAR LA MATRIZ DE CONFUSIÓN
        ================================================================================
        
        La matriz de confusión muestra:
        - Diagonal principal: Predicciones correctas
        - Fuera de la diagonal: Errores de clasificación
        
        INTERPRETACIÓN:
        ---------------
        - Valores altos en la diagonal = buen rendimiento
        - Valores fuera de la diagonal = confusiones entre clases
        """
        plt.figure(figsize=(10, 8))
        
        # Crear heatmap de la matriz de confusión
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, 
                   yticklabels=self.class_names,
                   cbar_kws={'label': 'Número de muestras'})
        
        plt.title(f'{title}\n', fontsize=16, fontweight='bold')
        plt.xlabel('Clase Predicha', fontsize=14)
        plt.ylabel('Clase Real', fontsize=14)
        
        # Añadir texto explicativo
        plt.figtext(0.5, 0.02, 
                   'Diagonal principal: Predicciones correctas\nFuera de diagonal: Errores de clasificación', 
                   ha='center', fontsize=10, style='italic')
        
        plt.tight_layout()
        plt.show()
    
    def plot_feature_importance(self):
        """
        ================================================================================
        VISUALIZAR LA IMPORTANCIA DE LAS CARACTERÍSTICAS
        ================================================================================
        
        Muestra los coeficientes de cada modelo para entender qué características
        son más importantes para distinguir cada clase.
        
        INTERPRETACIÓN:
        ---------------
        - Coeficientes positivos: La característica aumenta la probabilidad de la clase
        - Coeficientes negativos: La característica disminuye la probabilidad de la clase
        - Magnitud: Qué tan importante es la característica
        """
        plt.figure(figsize=(15, 5))
        
        for i, class_name in enumerate(self.class_names):
            plt.subplot(1, 3, i+1)
            
            # Obtener coeficientes del modelo
            coefficients = self.models[class_name].coef_
            
            # Crear gráfico de barras
            bars = plt.bar(self.feature_names, coefficients, 
                          color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
            
            # Personalizar gráfico
            plt.title(f'Importancia de Características\n{class_name}', 
                     fontsize=12, fontweight='bold')
            plt.xlabel('Características', fontsize=10)
            plt.ylabel('Valor del Coeficiente', fontsize=10)
            plt.xticks(rotation=45)
            plt.grid(axis='y', alpha=0.3)
            
            # Añadir valores en las barras
            for bar, coef in zip(bars, coefficients):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{coef:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.suptitle('Análisis de Importancia de Características por Clase', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def plot_data_distribution(self, X, y):
        """
        ================================================================================
        VISUALIZAR LA DISTRIBUCIÓN DE LOS DATOS
        ================================================================================
        
        Crea visualizaciones que muestran:
        1. Pairplot: Relaciones entre todas las características
        2. Matriz de correlación: Correlaciones entre características
        
        ESTAS GRÁFICAS AYUDAN A:
        ------------------------
        - Entender la separabilidad de las clases
        - Identificar características más discriminativas
        - Detectar patrones en los datos
        """
        # Crear DataFrame para facilitar el plotting
        df = pd.DataFrame(X, columns=self.feature_names)
        df['species'] = self.label_encoder.inverse_transform(y)
        
        # 1. PAIRPLOT - Relaciones entre características
        print("📊 Generando pairplot de características...")
        plt.figure(figsize=(15, 12))
        
        # Crear pairplot personalizado
        g = sns.pairplot(df, hue='species', diag_kind='hist', 
                        palette=['#FF6B6B', '#4ECDC4', '#45B7D1'],
                        plot_kws={'alpha': 0.7, 's': 50})
        
        g.fig.suptitle('Análisis de Relaciones entre Características del Dataset Iris', 
                      y=1.02, fontsize=16, fontweight='bold')
        
        # Personalizar leyenda
        g._legend.set_title('Especies de Iris')
        for text in g._legend.texts:
            text.set_fontsize(12)
        
        plt.show()
        
        # 2. MATRIZ DE CORRELACIÓN
        print("📊 Generando matriz de correlación...")
        plt.figure(figsize=(10, 8))
        
        # Calcular matriz de correlación
        correlation_matrix = df[self.feature_names].corr()
        
        # Crear heatmap
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='RdBu_r', 
                   center=0, square=True, linewidths=0.5,
                   cbar_kws={"shrink": 0.8})
        
        plt.title('Matriz de Correlación entre Características\n', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('Características', fontsize=12)
        plt.ylabel('Características', fontsize=12)
        
        # Añadir interpretación
        plt.figtext(0.5, 0.02, 
                   'Valores cercanos a 1: Correlación positiva fuerte\nValores cercanos a -1: Correlación negativa fuerte\nValores cercanos a 0: Poca correlación', 
                   ha='center', fontsize=10, style='italic')
        
        plt.tight_layout()
        plt.show()
    
    def plot_decision_boundaries_2d(self, X, y):
        """
        ================================================================================
        VISUALIZAR FRONTERAS DE DECISIÓN (PROYECCIÓN 2D)
        ================================================================================
        
        Muestra las fronteras de decisión del modelo usando solo las primeras
        dos características (longitud y ancho del sépalo).
        
        IMPORTANTE:
        -----------
        Esta es una proyección 2D de un modelo 4D. Las fronteras reales
        son más complejas en el espacio completo de 4 dimensiones.
        """
        print("📊 Generando fronteras de decisión 2D...")
        
        # Usar solo las primeras dos características para visualización 2D
        X_2d = X[:, :2]
        
        # Crear malla de puntos para las fronteras
        h = 0.02  # Tamaño del paso
        x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
        y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                           np.arange(y_min, y_max, h))
        
        # Hacer predicciones en la malla
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        # Completar con ceros para las otras características
        mesh_full = np.zeros((mesh_points.shape[0], 4))
        mesh_full[:, :2] = mesh_points
        
        # Predecir clases para todos los puntos de la malla
        Z = self.predict(mesh_full)[0]
        Z = Z.reshape(xx.shape)
        
        # Crear visualización
        plt.figure(figsize=(12, 10))
        
        # Dibujar fronteras de decisión
        plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
        
        # Dibujar puntos de datos reales
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        for i, class_name in enumerate(self.class_names):
            mask = y == i
            plt.scatter(X_2d[mask, 0], X_2d[mask, 1], 
                       c=colors[i], label=class_name, 
                       edgecolors='black', linewidth=0.5, s=60, alpha=0.8)
        
        # Personalizar gráfico
        plt.xlabel('Longitud del Sépalo (cm)', fontsize=12)
        plt.ylabel('Ancho del Sépalo (cm)', fontsize=12)
        plt.title('Fronteras de Decisión del Modelo\n(Proyección 2D usando Sépalo)', 
                 fontsize=14, fontweight='bold')
        plt.legend(title='Especies de Iris', loc='upper right')
        plt.colorbar(label='Clase Predicha')
        
        # Añadir nota explicativa
        plt.figtext(0.5, 0.02, 
                   'Nota: Esta es una proyección 2D. El modelo real usa 4 características.\nLas fronteras reales son más complejas en el espacio 4D completo.', 
                   ha='center', fontsize=10, style='italic')
        
        plt.tight_layout()
        plt.show()
    
    def print_detailed_results(self, results, dataset_name="Dataset"):
        """
        ================================================================================
        IMPRIMIR RESULTADOS DETALLADOS DE LA EVALUACIÓN
        ================================================================================
        
        Muestra un reporte completo y detallado de todas las métricas de evaluación.
        """
        print(f"\n{'='*80}")
        print(f"RESULTADOS DETALLADOS - {dataset_name.upper()}")
        print(f"{'='*80}")
        
        # Accuracy general
        accuracy = results['accuracy']
        print(f"\n🎯 PRECISIÓN GENERAL: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Reporte de clasificación detallado
        print(f"\n📊 REPORTE DE CLASIFICACIÓN DETALLADO:")
        print(f"{'-'*60}")
        report = results['classification_report']
        
        # Métricas por clase
        for class_name in self.class_names:
            if class_name in report:
                metrics = report[class_name]
                print(f"\n🌺 {class_name.upper()}:")
                print(f"   📈 Precisión: {metrics['precision']:.4f} - De las predicciones positivas, {metrics['precision']*100:.1f}% son correctas")
                print(f"   📈 Recall:    {metrics['recall']:.4f} - De los casos reales, {metrics['recall']*100:.1f}% fueron detectados")
                print(f"   📈 F1-Score:  {metrics['f1-score']:.4f} - Media armónica entre precisión y recall")
                print(f"   📈 Soporte:   {metrics['support']:.0f} - Número de muestras de esta clase")
        
        # Promedios
        print(f"\n📊 PROMEDIOS:")
        print(f"   📈 Precisión Macro: {report['macro avg']['precision']:.4f}")
        print(f"   📈 Recall Macro:    {report['macro avg']['recall']:.4f}")
        print(f"   📈 F1-Score Macro:  {report['macro avg']['f1-score']:.4f}")
        print(f"   📈 Precisión Ponderada: {report['weighted avg']['precision']:.4f}")
        print(f"   📈 Recall Ponderado:    {report['weighted avg']['recall']:.4f}")
        print(f"   📈 F1-Score Ponderado:  {report['weighted avg']['f1-score']:.4f}")
        
        # Matriz de confusión
        print(f"\n🔢 MATRIZ DE CONFUSIÓN:")
        print(f"{'-'*60}")
        cm = results['confusion_matrix']
        
        # Encabezado
        print("Real\\Predicho", end="")
        for class_name in self.class_names:
            print(f"\t{class_name[:8]}", end="")
        print()
        
        # Filas de la matriz
        for i, class_name in enumerate(self.class_names):
            print(f"{class_name[:8]}", end="")
            for j in range(len(self.class_names)):
                print(f"\t\t{cm[i][j]}", end="")
            print()
        
        # Interpretación de la matriz
        print(f"\n💡 INTERPRETACIÓN:")
        print(f"   ✅ Diagonal principal: Predicciones correctas")
        print(f"   ❌ Fuera de diagonal: Errores de clasificación")
        for i in range(len(self.class_names)):
            correct = cm[i][i]
            total = cm[i].sum()
            print(f"   📊 {self.class_names[i]}: {correct}/{total} correctas ({correct/total*100:.1f}%)")


def main():
    """
    ================================================================================
    FUNCIÓN PRINCIPAL - EJECUTAR CLASIFICACIÓN COMPLETA
    ================================================================================
    
    Esta función ejecuta todo el pipeline de clasificación:
    1. Cargar datos
    2. Preprocesar
    3. Dividir en entrenamiento y prueba
    4. Entrenar modelos
    5. Evaluar rendimiento
    6. Generar visualizaciones
    7. Mostrar resultados detallados
    """
    print("="*80)
    print("🌺 CLASIFICACIÓN DEL DATASET IRIS USANDO REGRESIÓN LINEAL 🌺")
    print("="*80)
    print("📋 Este programa clasifica flores de iris en 3 especies usando regresión lineal")
    print("🔬 Método: One-vs-Rest con regresión lineal estándar")
    print("="*80)
    
    # ================================================================================
    # 1. INICIALIZAR CLASIFICADOR
    # ================================================================================
    print("\n🚀 PASO 1: INICIALIZANDO CLASIFICADOR")
    print("-" * 50)
    classifier = IrisLinearClassifier(use_regularization=False)
    
    # ================================================================================
    # 2. CARGAR Y PREPROCESAR DATOS
    # ================================================================================
    print("\n🚀 PASO 2: CARGANDO Y PREPROCESANDO DATOS")
    print("-" * 50)
    X, y = classifier.load_data('iris.data')
    X_scaled, y_encoded = classifier.preprocess_data(X, y)
    
    # ================================================================================
    # 3. DIVIDIR DATOS EN ENTRENAMIENTO Y PRUEBA
    # ================================================================================
    print("\n🚀 PASO 3: DIVIDIENDO DATOS EN ENTRENAMIENTO Y PRUEBA")
    print("-" * 50)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
    )
    
    print(f"✅ División completada:")
    print(f"   📊 Conjunto de entrenamiento: {X_train.shape[0]} muestras")
    print(f"   📊 Conjunto de prueba: {X_test.shape[0]} muestras")
    print(f"   📊 Características: {X_train.shape[1]}")
    
    # ================================================================================
    # 4. ENTRENAR MODELOS
    # ================================================================================
    print("\n🚀 PASO 4: ENTRENANDO MODELOS DE REGRESIÓN LINEAL")
    print("-" * 50)
    classifier.train_models(X_train, y_train)
    
    # ================================================================================
    # 5. EVALUAR RENDIMIENTO
    # ================================================================================
    print("\n🚀 PASO 5: EVALUANDO RENDIMIENTO DEL MODELO")
    print("-" * 50)
    
    # Evaluar en conjunto de entrenamiento
    print("\n📊 EVALUACIÓN EN CONJUNTO DE ENTRENAMIENTO:")
    train_results = classifier.evaluate(X_train, y_train)
    classifier.print_detailed_results(train_results, "Entrenamiento")
    
    # Evaluar en conjunto de prueba
    print("\n📊 EVALUACIÓN EN CONJUNTO DE PRUEBA:")
    test_results = classifier.evaluate(X_test, y_test)
    classifier.print_detailed_results(test_results, "Prueba")
    
    # ================================================================================
    # 6. GENERAR VISUALIZACIONES
    # ================================================================================
    print("\n🚀 PASO 6: GENERANDO VISUALIZACIONES")
    print("-" * 50)
    print("📊 Generando gráficas de análisis...")
    
    # Distribución de datos
    classifier.plot_data_distribution(X, y_encoded)
    
    # Matrices de confusión
    classifier.plot_confusion_matrix(train_results['confusion_matrix'], 
                                   "Matriz de Confusión - Entrenamiento")
    classifier.plot_confusion_matrix(test_results['confusion_matrix'], 
                                   "Matriz de Confusión - Prueba")
    
    # Importancia de características
    classifier.plot_feature_importance()
    
    # Fronteras de decisión
    classifier.plot_decision_boundaries_2d(X, y_encoded)
    
    # ================================================================================
    # 7. RESUMEN FINAL
    # ================================================================================
    print("\n" + "="*80)
    print("📋 RESUMEN FINAL DEL EXPERIMENTO")
    print("="*80)
    
    train_accuracy = train_results['accuracy']
    test_accuracy = test_results['accuracy']
    overfitting = train_accuracy - test_accuracy
    
    print(f"\n🎯 RESULTADOS PRINCIPALES:")
    print(f"   📈 Precisión de Entrenamiento: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
    print(f"   📈 Precisión de Prueba: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"   📈 Sobreajuste: {overfitting:.4f} ({overfitting*100:.2f}%)")
    
    print(f"\n🔍 ANÁLISIS DE RENDIMIENTO:")
    if overfitting > 0.1:
        print(f"   ⚠️  El modelo muestra sobreajuste significativo")
    elif overfitting < -0.05:
        print(f"   ⚠️  El modelo podría estar subajustado")
    else:
        print(f"   ✅ El modelo tiene un balance adecuado entre entrenamiento y prueba")
    
    print(f"\n💡 CONCLUSIONES:")
    print(f"   🌺 Iris-setosa: Fácilmente separable (linealmente separable)")
    print(f"   🌺 Iris-versicolor y virginica: Más difíciles de separar")
    print(f"   📊 La regresión lineal funciona razonablemente bien para este problema")
    print(f"   🔬 Para mejores resultados, considerar regresión logística o SVM")
    
    print(f"\n✅ EXPERIMENTO COMPLETADO EXITOSAMENTE")
    print("="*80)
    
    return classifier, train_results, test_results


if __name__ == "__main__":
    # Ejecutar el experimento completo
    classifier, train_results, test_results = main()
