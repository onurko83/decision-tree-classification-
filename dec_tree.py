import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Sayfa başlığı
st.title('İki Özellikli Binary Classification Veri Görselleştirme')

# Parametre seçimi için sidebar
st.sidebar.header('Decision Tree Classification')

# Ana sayfa ortasında parametre seçimi için container
with st.container():
    st.markdown("---")
    st.subheader('Veri Oluşturma Parametreleri')
    
    # İki sütunlu layout oluştur
    col1, col2 = st.columns(2)

    # Sol sütun
    with col1:
        # n_samples parametresi
        n_samples = st.slider(
            'Örnek Sayısı (n_samples):',
            min_value=10,
            max_value=1000,
            value=100,
            step=20,
            help='Veri setindeki toplam örnek sayısını belirler'
        )
        
        # centers parametresi
        center_options = [[[-2, -2], [2, 2]], [[-3, -3], [3, 3]], [[-1, -1], [1, 1]]]
        center_index = st.slider(
            'Merkez Konumları (centers):',
            min_value=0,
            max_value=len(center_options)-1,
            value=0,
            help='Küme merkezlerinin konumları'
        )
        centers = center_options[center_index]

    # Sağ sütun
    with col2:
        # cluster_std parametresi
        cluster_std = st.slider(
            'Küme Standart Sapması (cluster_std):',
            min_value=0.1,
            max_value=10.0,
            value=1.0,
            step=0.1,
            help='Kümelerin dağılım genişliği. Düşük değerler daha sıkı kümeler oluşturur'
        )

# Yapay veri oluşturma
X, y = make_blobs(
    n_samples=n_samples,
    centers=centers,
    n_features=2,
    cluster_std=cluster_std,
    random_state=42
)

# DataFrame oluşturma
df = pd.DataFrame(X, columns=['X1', 'X2'])
df['target'] = y

# Veriyi gösterme
st.write('Oluşturulan Veri:')
#st.dataframe(df)

# Görselleştirme
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='X1', y='X2', hue='target', palette='Set1', s=50)
plt.title('İki Özellikli Veri Dağılımı')
st.pyplot(plt)

# Özellik seçimi için selectbox
feature = st.selectbox('Özellik Seçin:', ['X1', 'X2'])

# Seçilen özelliğe göre verileri sırala
sorted_values = np.sort(df[feature].values)
thresholds = []

# Ardışık data pointlerin orta noktalarını hesapla
for i in range(len(sorted_values)-1):
    threshold = (sorted_values[i] + sorted_values[i+1]) / 2
    thresholds.append(threshold)

# Threshold seçimi için selectbox (sadece orta nokta değerleri)
if thresholds:
    selected_threshold = st.selectbox(
        'Threshold Seçin:',
        options=thresholds,
        format_func=lambda x: f'{x:.2f}',
        help='Threshold değerini seçin (ardışık değerlerin orta noktaları)'
    )
else:
    selected_threshold = 0

# Seçilen özellik için scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='X1', y='X2', hue='target', palette='Set1', s=50)

# Threshold çizgisini ekle
if feature == 'X1':
    plt.axvline(x=selected_threshold, color='red', linestyle='--', label=f'Threshold: {selected_threshold:.2f}')
else:
    plt.axhline(y=selected_threshold, color='red', linestyle='--', label=f'Threshold: {selected_threshold:.2f}')

plt.title(f'Threshold ile İki Özellikli Veri Dağılımı')
plt.legend()
st.pyplot(plt)

# Entropy hesaplama fonksiyonu
def calculate_entropy(y):
    if len(y) == 0:
        return 0
    # Sınıf olasılıklarını hesapla
    probabilities = np.bincount(y) / len(y)
    # Sıfır olasılıklarını filtrele ve entropy'yi hesapla
    entropy = -np.sum([p * np.log2(p) for p in probabilities if p > 0])
    return entropy 
# Entropy grafiği için container
entropy_graph = st.empty()

# Seçilen threshold'a göre entropy değerlerini hesapla ve grafiği çiz
with entropy_graph.container():
    st.write('### Entropy Değerleri Grafiği')
    
    # Threshold değerlerini ve entropy'leri hesapla
    threshold_entropies = []
    threshold_values = []
    
    for threshold in thresholds:
        # Threshold'a göre veriyi böl
        if feature == 'X1':
            left_mask = df[feature] <= threshold
            right_mask = df[feature] > threshold
        else:  # feature == 'X2'
            left_mask = df[feature] <= threshold
            right_mask = df[feature] > threshold
        
        left_labels = df[left_mask]['target'].values
        right_labels = df[right_mask]['target'].values
        
        # Sol ve sağ entropy'leri hesapla
        left_entropy = calculate_entropy(left_labels)
        right_entropy = calculate_entropy(right_labels)
        
        # Ağırlıklı entropy
        total_samples = len(df)
        weighted_entropy = (len(left_labels) / total_samples) * left_entropy + (len(right_labels) / total_samples) * right_entropy
        
        threshold_entropies.append(weighted_entropy)
        threshold_values.append(threshold)
    
    # Entropy grafiğini çiz
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(threshold_values, threshold_entropies, 'b-', linewidth=2, label='Ağırlıklı Entropy')
    
    # Seçilen threshold'u işaretle
    if thresholds:
        selected_index = min(range(len(thresholds)), key=lambda i: abs(thresholds[i] - selected_threshold))
        ax.scatter(selected_threshold, threshold_entropies[selected_index], color='red', s=100, zorder=5, label=f'Seçilen Threshold: {selected_threshold:.2f}')
    
    ax.set_xlabel('Threshold Değeri')
    ax.set_ylabel('Ağırlıklı Entropy')
    ax.set_title(f'{feature} Özelliği için Entropy Grafiği')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Seçilen threshold'un entropy değerini göster
    if thresholds:
        st.write(f"**Seçilen Threshold ({selected_threshold:.2f}) için Ağırlıklı Entropy:** {threshold_entropies[selected_index]:.3f}")
    
    st.pyplot(fig)

# Entropy'nin genel grafiği
with st.container():
    st.write('### Entropy Grafiği')
    
    # 0-1 aralığında slider
    p_value = st.slider('Label 1 Olasılığı (p)', 0.0, 1.0, 0.5, 0.01)
    
    # Entropy formülü
    st.write('#### Entropy Formülü:')
    st.latex(r'''Entropy(p) = -p \log_2(p) - (1-p) \log_2(1-p)''')
    
    # Seçilen olasılık için entropy hesaplama
    if p_value == 0 or p_value == 1:
        entropy_value = 0.0
    else:
        entropy_value = -p_value * np.log2(p_value) - (1-p_value) * np.log2(1-p_value)
    
    # Formül hesaplama gösterimi
    if p_value > 0 and p_value < 1:
        st.write('#### Hesaplama:')
        st.latex(fr'''Entropy({p_value:.3f}) = -{p_value:.3f} \times \log_2({p_value:.3f}) - {1-p_value:.3f} \times \log_2({1-p_value:.3f}) = {entropy_value:.3f}''')
    else:
        st.write('#### Hesaplama:')
        st.latex(fr'''Entropy({p_value:.3f}) = 0.000''')
    
    # Entropy grafiği
    p_range = np.linspace(0.001, 0.999, 1000)
    entropy_range = -p_range * np.log2(p_range) - (1-p_range) * np.log2(1-p_range)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(p_range, entropy_range, 'b-', linewidth=2, label='Entropy')
    ax.scatter(p_value, entropy_value, color='red', s=100, zorder=5, label=f'Seçilen nokta: p={p_value:.3f}')
    
    ax.set_xlabel('Label 1 Olasılığı (p)')
    ax.set_ylabel('Entropy')
    ax.set_title('Entropy Genel Grafiği')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.5)
    
    st.pyplot(fig)

# Threshold'a göre veriyi bölme
if feature == 'X1':
    left_mask = df[feature] <= selected_threshold
    right_mask = df[feature] > selected_threshold
else: # feature == 'X2'
    left_mask = df[feature] <= selected_threshold
    right_mask = df[feature] > selected_threshold

# Sol ve sağ dallardaki etiketleri al
left_labels = df[left_mask]['target'].values
right_labels = df[right_mask]['target'].values

# Gerekli tüm entropy değerlerini ve bilgi kazancını hesapla
parent_entropy = calculate_entropy(df['target'].values)
left_entropy = calculate_entropy(left_labels)
right_entropy = calculate_entropy(right_labels)

total_samples = len(df)
weighted_entropy = (len(left_labels) / total_samples) * left_entropy + (len(right_labels) / total_samples) * right_entropy
information_gain = parent_entropy - weighted_entropy

# --- Adım Adım Formül Gösterimi ---
st.write('### Bilgi Kazancı Hesaplaması - Adım Adım')

# 1. Ana Entropy Hesaplama
st.write('#### 1. Ana Entropy Hesaplama:')
st.latex(r'''Entropy(S) = -\sum_{i=1}^{c} p_i \log_2(p_i)''')

classes_parent, counts_parent = np.unique(df['target'].values, return_counts=True)
total_parent = len(df['target'].values)
if total_parent > 0:
    latex_terms = [rf"\frac{{{c}}}{{{total_parent}}} \times \log_2\left(\frac{{{c}}}{{{total_parent}}}\right)" for c in counts_parent]
    latex_string = r"Entropy(Ana) = -\left[" + " + ".join(latex_terms) + rf"\right] = {parent_entropy:.3f}"
    st.latex(latex_string)

# 2. Sol Dal Entropy Hesaplama
st.write('#### 2. Sol Dal Entropy Hesaplama:')
st.latex(r'''Entropy(S) = -\sum_{i=1}^{c} p_i \log_2(p_i)''')

if len(left_labels) > 0:
    classes_left, counts_left = np.unique(left_labels, return_counts=True)
    total_left = len(left_labels)
    if len(counts_left) == 1:
        # Tek sınıf varsa entropy 0'dır, formülü doğru göster
        p = counts_left[0] / total_left
        latex_string_left = rf"Entropy(Sol) = -\left[\frac{{{counts_left[0]}}}{{{total_left}}} \times \log_2\left(\frac{{{counts_left[0]}}}{{{total_left}}}\right)\right] = -[{p:.3f} \times \log_2({p:.3f})] = 0.000"
    else:
        latex_terms_left = [rf"\frac{{{c}}}{{{total_left}}} \times \log_2\left(\frac{{{c}}}{{{total_left}}}\right)" for c in counts_left]
        latex_string_left = r"Entropy(Sol) = -\left[" + " + ".join(latex_terms_left) + rf"\right] = {left_entropy:.3f}"
    st.latex(latex_string_left)
else:
    st.latex(r"Entropy(Sol) = 0.000")

# 3. Sağ Dal Entropy Hesaplama
st.write('#### 3. Sağ Dal Entropy Hesaplama:')
st.latex(r'''Entropy(S) = -\sum_{i=1}^{c} p_i \log_2(p_i)''')

if len(right_labels) > 0:
    classes_right, counts_right = np.unique(right_labels, return_counts=True)
    total_right = len(right_labels)
    if len(counts_right) == 1:
        # Tek sınıf varsa entropy 0'dır, formülü doğru göster
        p = counts_right[0] / total_right
        latex_string_right = rf"Entropy(Sağ) = -\left[\frac{{{counts_right[0]}}}{{{total_right}}} \times \log_2\left(\frac{{{counts_right[0]}}}{{{total_right}}}\right)\right] = -[{p:.3f} \times \log_2({p:.3f})] = 0.000"
    else:
        latex_terms_right = [rf"\frac{{{c}}}{{{total_right}}} \times \log_2\left(\frac{{{c}}}{{{total_right}}}\right)" for c in counts_right]
        latex_string_right = r"Entropy(Sağ) = -\left[" + " + ".join(latex_terms_right) + rf"\right] = {right_entropy:.3f}"
    st.latex(latex_string_right)
else:
    st.latex(r"Entropy(Sağ) = 0.000")

# 4. Ağırlıklı Entropy Hesaplama
st.write('#### 4. Ağırlıklı Entropy Hesaplama:')
st.latex(r'''Weighted\_Entropy = \frac{n_{left}}{n_{total}} \times Entropy(Sol) + \frac{n_{right}}{n_{total}} \times Entropy(Sağ)''')
latex_calc_weighted = rf"Weighted\_Entropy = \frac{{{len(left_labels)}}}{{{total_samples}}} \times {left_entropy:.3f} + \frac{{{len(right_labels)}}}{{{total_samples}}} \times {right_entropy:.3f} = {weighted_entropy:.3f}"
st.latex(latex_calc_weighted)

# 5. Information Gain Hesaplama
st.write('#### 5. Information Gain Hesaplama:')
st.latex(r'''Information\_Gain = Entropy(Ana) - Weighted\_Entropy''')
latex_calc_ig = rf"Information\_Gain = {parent_entropy:.3f} - {weighted_entropy:.3f} = {information_gain:.3f}"
st.latex(latex_calc_ig)


# Sidebar'da Decision Tree parametreleri
with st.sidebar:
    st.write('## Decision Tree Parametreleri')
    
    # Temel parametreler
    st.write('### Temel Parametreler')
    max_depth = st.slider('Maksimum Derinlik (max_depth)', 1, 20, 5)
    st.caption('Ağacın maksimum derinliği. Arttırılırsa daha karmaşık model, azaltılırsa daha basit model.')
    
    min_samples_split = st.slider('Minimum Split Örneği (min_samples_split)', 2, 20, 2)
    st.caption('Bir düğümün bölünmesi için gereken minimum örnek sayısı. Arttırılırsa daha az bölünme.')
    
    min_samples_leaf = st.slider('Minimum Yaprak Örneği (min_samples_leaf)', 1, 10, 1)
    st.caption('Bir yaprak düğümünde bulunması gereken minimum örnek sayısı. Arttırılırsa daha az yaprak.')
    
    criterion = st.selectbox('Kriter (criterion)', ['gini', 'entropy', 'log_loss'])
    st.caption('Bölünme kriteri. Gini daha hızlı, Entropy daha bilgilendirici.')
    
    # Gelişmiş parametreler
    st.write('### Gelişmiş Parametreler')
    max_features = st.selectbox('Maksimum Özellik (max_features)', ['sqrt', 'log2', None, 1, 2])
    st.caption('Her bölünmede değerlendirilecek maksimum özellik sayısı. sqrt/log2/None seçenekleri.')
    
    random_state = st.slider('Random State', 0, 100, 42)
    st.caption('Rastgelelik için seed değeri. Sabit değer tekrarlanabilir sonuçlar sağlar.')
    
    splitter = st.selectbox('Splitter', ['best', 'random'])
    st.caption('Bölünme stratejisi. Best en iyi bölünmeyi, random rastgele bölünme seçer.')
    
    # Ek parametreler
    st.write('### Ek Parametreler')
    max_leaf_nodes = st.slider('Maksimum Yaprak Düğümü (max_leaf_nodes)', 2, 50, None)
    st.caption('Maksimum yaprak düğüm sayısı. Arttırılırsa daha fazla yaprak.')
    
    min_weight_fraction_leaf = st.slider('Min Ağırlık Kesri (min_weight_fraction_leaf)', 0.0, 0.5, 0.0, 0.01)
    st.caption('Yaprak düğümünde bulunması gereken minimum ağırlık kesri.')
    
    ccp_alpha = st.slider('CCP Alpha (ccp_alpha)', 0.0, 1.0, 0.0, 0.01)
    st.caption('Cost complexity pruning parametresi. Arttırılırsa daha fazla budama.')
    
    # Class weight
    class_weight = st.selectbox('Class Weight', ['None', 'balanced'])
    st.caption('Sınıf ağırlıkları. Balanced dengesiz veri setleri için otomatik ağırlık.')

# Decision Tree eğitimi
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# max_features değerini düzelt
if max_features in ['sqrt', 'log2', None]:
    max_features_final = max_features
else:
    max_features_final = int(max_features)

# class_weight değerini düzelt
if class_weight == 'None':
    class_weight_final = None
else:
    class_weight_final = class_weight

# Model oluştur ve eğit
dt_model = DecisionTreeClassifier(
    max_depth=max_depth,
    min_samples_split=min_samples_split,
    min_samples_leaf=min_samples_leaf,
    criterion=criterion,
    max_features=max_features_final,
    random_state=random_state,
    splitter=splitter,
    max_leaf_nodes=max_leaf_nodes if max_leaf_nodes != None else None,
    min_weight_fraction_leaf=min_weight_fraction_leaf,
    ccp_alpha=ccp_alpha,
    class_weight=class_weight_final
)

feature_names=['X1','X2']
class_names=['0','1']

# Modeli eğit
dt_model.fit(X, y)

# Ağaç derinliğini kontrol et ve göster
actual_depth = dt_model.get_depth()
st.write(f'**Gerçek Ağaç Derinliği: {actual_depth}**')
st.write(f'**Maksimum Ağaç Derinliği: {max_depth}**')

# Decision Tree görselleştirme
st.write('## Decision Tree Ağacı')

# Matplotlib figure oluştur - daha büyük boyut
fig, ax = plt.subplots(figsize=(60, 50))

# Ağacı çiz - daha büyük font ve daha iyi görünüm
plot_tree(dt_model, 
          feature_names=feature_names,
          class_names=class_names,
          filled=True,
          rounded=True,
          fontsize=60,  # Font boyutunu artırdım
          ax=ax,
          node_ids=True,  # Düğüm ID'lerini göster
          proportion=True)  # Oranları göster

# Layout'u düzenle
plt.tight_layout(pad=2.0)  # Padding artırdım
st.pyplot(fig, use_container_width=True)  # Container genişliğini kullan

# Model performansı
from sklearn.metrics import accuracy_score, classification_report

# Tahmin yap
y_pred = dt_model.predict(X)

# Doğruluk skoru
accuracy = accuracy_score(y, y_pred)
st.write(f'**Model Doğruluğu: {accuracy:.3f}**')

# Sınıflandırma raporu
st.write('## Sınıflandırma Raporu')
st.text(classification_report(y, y_pred, target_names=class_names))


# Görselleştirme
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='X1', y='X2', hue='target', palette='Set1', s=100, linewidth=0)
plt.title('İki Özellikli Veri Dağılımı')
st.pyplot(plt)

# Decision boundary plot - renklendirmeli
st.write('## Decision Boundary Plot (Renklendirmeli)')

# Mesh grid oluştur
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))

# Mesh grid üzerinde tahmin yap
Z = dt_model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot oluştur
fig, ax = plt.subplots(figsize=(10, 8))

# Decision boundary'yi çiz - daha belirgin renkler kullan
contour = ax.contourf(xx, yy, Z, alpha=0.4, colors=['#FF6B6B', '#4ECDC4'], levels=[0, 0.5, 1])

# Veri noktalarını çiz - edge olmadan
scatter = sns.scatterplot(data=df, x='X1', y='X2', hue='target', palette='Set1', s=120, 
                         linewidth=0, ax=ax)

# Plot ayarları
ax.set_xlabel('X1', fontsize=12)
ax.set_ylabel('X2', fontsize=12)
ax.set_title('Decision Tree Decision Boundary', fontsize=14, fontweight='bold')

plt.tight_layout()
st.pyplot(fig, use_container_width=True)



# teorik algoritma nasıl çalışıyor, parameterelr ne işe yarıyor.

# geçmiş yarışma seç data problem var metrik yüksek performans almaya çalış.
# kaggle playground git eski yarışma seç ve datayı alıp sıfırdan mdoel geliştir ve public private skoru gözleme private skoru arttırmaya çalış.

# tabular regression classification time series, tablo datası 
