import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# --- تحميل البيانات ---
data = [
  {"date": "5/9/2025, 11:53:50 PM", "duration": 11, "jumps": 5, "score": 0, "obstaclesPassed": 0},
  {"date": "5/9/2025, 11:54:16 PM", "duration": 14, "jumps": 10, "score": 1, "obstaclesPassed": 1},
  {"date": "5/9/2025, 11:55:19 PM", "duration": 67, "jumps": 40, "score": 10, "obstaclesPassed": 12},
  {"date": "5/9/2025, 11:56:46 PM", "duration": 153, "jumps": 100, "score": 30, "obstaclesPassed": 35},
  {"date": "5/10/2025, 1:12:26 PM", "duration": 20, "jumps": 8, "score": 1, "obstaclesPassed": 2},
  {"date": "5/11/2025, 9:12:22 PM", "duration": 20, "jumps": 9, "score": 2, "obstaclesPassed": 3},
  {"date": "5/11/2025, 9:13:01 PM", "duration": 59, "jumps": 55, "score": 12, "obstaclesPassed": 14},
  {"date": "5/11/2025, 9:15:31 PM", "duration": 209, "jumps": 120, "score": 40, "obstaclesPassed": 42},
  {"date": "5/11/2025, 9:16:00 PM", "duration": 180, "jumps": 85, "score": 28, "obstaclesPassed": 30},
  {"date": "5/11/2025, 9:20:00 PM", "duration": 90, "jumps": 50, "score": 15, "obstaclesPassed": 17}
]

df = pd.DataFrame(data)

# --- إضافة عمود label ---
df['label'] = df['score'].apply(lambda x: 1 if x >= 10 else 0)

st.title("لوحة تحكم تحليل بيانات اللعبة")

st.header("عرض البيانات")
st.dataframe(df)

# --- التحليل الاستكشافي ---
st.header("تحليل استكشافي")

fig, ax = plt.subplots()
sns.scatterplot(data=df, x='duration', y='jumps', ax=ax)
plt.title("العلاقة بين مدة اللعب وعدد القفزات")
st.pyplot(fig)

fig2, ax2 = plt.subplots()
sns.scatterplot(data=df, x='obstaclesPassed', y='score', ax=ax2)
plt.title("العلاقة بين العقبات والنقاط")
st.pyplot(fig2)

# --- K-Means Clustering ---
st.header("تصنيف اللاعبين - K-Means")

features = df[['duration', 'jumps', 'score', 'obstaclesPassed']]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(scaled_features)

st.write("البيانات بعد التصنيف:")
st.dataframe(df[['duration', 'jumps', 'score', 'obstaclesPassed', 'cluster']])

fig3, ax3 = plt.subplots()
sns.scatterplot(data=df, x='duration', y='score', hue='cluster', palette='Set1', s=100, ax=ax3)
plt.title("مجموعات اللاعبين حسب K-Means")
st.pyplot(fig3)

# --- Decision Tree Classifier ---
st.header("تصنيف أداء الجلسات - Decision Tree")

X = df[['duration', 'jumps', 'obstaclesPassed']]
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
st.write(f"دقة نموذج Decision Tree على بيانات الاختبار: {accuracy:.2f}")

fig4, ax4 = plt.subplots(figsize=(12,6))
plot_tree(model, feature_names=X.columns, class_names=['فشل', 'نجاح'], filled=True, ax=ax4)
st.pyplot(fig4)

# --- التوصيات التلقائية ---
st.header("التوصيات التلقائية")

total_sessions = len(df)
failures = df[df['label'] == 0]
failure_rate = len(failures) / total_sessions

short_failures = failures[failures['duration'] < 30]
short_failure_rate = len(short_failures) / len(failures) if len(failures) > 0 else 0

if failure_rate > 0.5:
    st.warning(f"- أكثر من {failure_rate:.0%} من الجلسات تنتهي بالفشل – يُنصح بمراجعة توازن اللعبة.")

if short_failure_rate > 0.5:
    st.warning(f"- {short_failure_rate:.0%} من حالات الفشل حدثت خلال أول 30 ثانية – يُنصح بإضافة تعليمات مبسطة.")

if df['jumps'].mean() > 50:
    st.info("- متوسط عدد القفزات مرتفع – تأكد من وضوح آلية القفز.")

if df['score'].max() - df['score'].mean() > 20:
    st.info("- هناك تباين كبير في النقاط – قد تكون بعض العقبات صعبة.")

if df['obstaclesPassed'].mean() < 10:
    st.info("- معظم اللاعبين لم يتجاوزوا عددًا كبيرًا من العقبات – ربما العقبات الأولى صعبة.")
