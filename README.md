# مشروع تحليل بيانات لعبة 🎮 - Game Analytics Project

---

## نظرة عامة
هذا المشروع يهدف إلى تحليل بيانات جلسات لعب من لعبة بسيطة باستخدام تقنيات تعلم الآلة (Machine Learning).  
الهدف هو فهم سلوك اللاعبين، تصنيفهم بناءً على أنماط لعبهم، وتقديم توصيات ذكية لتحسين تصميم اللعبة وتجربة المستخدم.

---

## محتويات المشروع

- **ملف التحليل الرئيسي:**  
  `Game_Analysis_HumanStyle_ipynb_shaymaa.ipynb`  
  يحتوي على تحليل البيانات، تطبيق خوارزميات K-Means وDecision Tree، واستخلاص التوصيات.

- **تطبيق واجهة المستخدم التفاعلية:**  
  `app.py`  
  تطبيق Streamlit لعرض البيانات والنتائج والتوصيات بشكل تفاعلي.

- **ملف المتطلبات:**  
  `requirements.txt`  
  يحتوي على قائمة بالمكتبات البرمجية اللازمة لتشغيل المشروع.

- **ملف التوثيق هذا:**  
  `README.md`

---

## التقنيات المستخدمة

- لغة البرمجة: Python  
- المكتبات:  
  - pandas  
  - matplotlib  
  - seaborn  
  - scikit-learn  
  - streamlit

---

## كيفية تشغيل المشروع

### تشغيل الـ Notebook (Google Colab)

1. افتح ملف `Game_Analysis_HumanStyle_ipynb_shaymaa.ipynb` في Google Colab.  
2. نفذ خلايا الكود خطوة بخطوة لمتابعة التحليل والنماذج والتوصيات.
3. هاي خطوات كيف عرضت النتائج بستخدام stramlit

### تشغيل تطبيق Streamlit محليًا

1. تأكد من تثبيت Python على جهازك.  
2. ثبّت المكتبات المطلوبة باستخدام الأمر:

   Microsoft Windows [Version 10.0.26100.3915]
(c) Microsoft Corporation. All rights reserved.

C:\Users\PAVILION>pip install scikit-learn
Collecting scikit-learn
  Downloading scikit_learn-1.6.1-cp313-cp313-win_amd64.whl.metadata (15 kB)
Requirement already satisfied: numpy>=1.19.5 in c:\users\pavilion\appdata\local\programs\python\python313\lib\site-packages (from scikit-learn) (2.2.5)
Collecting scipy>=1.6.0 (from scikit-learn)
  Downloading scipy-1.15.3-cp313-cp313-win_amd64.whl.metadata (60 kB)
Collecting joblib>=1.2.0 (from scikit-learn)
  Downloading joblib-1.5.0-py3-none-any.whl.metadata (5.6 kB)
Collecting threadpoolctl>=3.1.0 (from scikit-learn)
  Downloading threadpoolctl-3.6.0-py3-none-any.whl.metadata (13 kB)
Downloading scikit_learn-1.6.1-cp313-cp313-win_amd64.whl (11.1 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 11.1/11.1 MB 8.1 MB/s eta 0:00:00
Downloading joblib-1.5.0-py3-none-any.whl (307 kB)
Downloading scipy-1.15.3-cp313-cp313-win_amd64.whl (41.0 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 41.0/41.0 MB 3.7 MB/s eta 0:00:00
Downloading threadpoolctl-3.6.0-py3-none-any.whl (18 kB)
Installing collected packages: threadpoolctl, scipy, joblib, scikit-learn
Successfully installed joblib-1.5.0 scikit-learn-1.6.1 scipy-1.15.3 threadpoolctl-3.6.0

[notice] A new release of pip is available: 25.0.1 -> 25.1.1
[notice] To update, run: python.exe -m pip install --upgrade pip

C:\Users\PAVILION>pip install streamlit pandas matplotlib seaborn
Requirement already satisfied: streamlit in c:\users\pavilion\appdata\local\programs\python\python313\lib\site-packages (1.45.1)
Requirement already satisfied: pandas in c:\users\pavilion\appdata\local\programs\python\python313\lib\site-packages (2.2.3)
Requirement already satisfied: matplotlib in c:\users\pavilion\appdata\local\programs\python\python313\lib\site-packages (3.10.3)
Requirement already satisfied: seaborn in c:\users\pavilion\appdata\local\programs\python\python313\lib\site-packages (0.13.2)
Requirement already satisfied: altair<6,>=4.0 in c:\users\pavilion\appdata\local\programs\python\python313\lib\site-packages (from streamlit) (5.5.0)
Requirement already satisfied: blinker<2,>=1.5.0 in c:\users\pavilion\appdata\local\programs\python\python313\lib\site-packages (from streamlit) (1.9.0)
Requirement already satisfied: cachetools<6,>=4.0 in c:\users\pavilion\appdata\local\programs\python\python313\lib\site-packages (from streamlit) (5.5.2)
Requirement already satisfied: click<9,>=7.0 in c:\users\pavilion\appdata\local\programs\python\python313\lib\site-packages (from streamlit) (8.2.0)
Requirement already satisfied: numpy<3,>=1.23 in c:\users\pavilion\appdata\local\programs\python\python313\lib\site-packages (from streamlit) (2.2.5)
Requirement already satisfied: packaging<25,>=20 in c:\users\pavilion\appdata\local\programs\python\python313\lib\site-packages (from streamlit) (24.2)
Requirement already satisfied: pillow<12,>=7.1.0 in c:\users\pavilion\appdata\local\programs\python\python313\lib\site-packages (from streamlit) (11.2.1)
Requirement already satisfied: protobuf<7,>=3.20 in c:\users\pavilion\appdata\local\programs\python\python313\lib\site-packages (from streamlit) (6.31.0)
Requirement already satisfied: pyarrow>=7.0 in c:\users\pavilion\appdata\local\programs\python\python313\lib\site-packages (from streamlit) (20.0.0)
Requirement already satisfied: requests<3,>=2.27 in c:\users\pavilion\appdata\local\programs\python\python313\lib\site-packages (from streamlit) (2.32.3)
Requirement already satisfied: tenacity<10,>=8.1.0 in c:\users\pavilion\appdata\local\programs\python\python313\lib\site-packages (from streamlit) (9.1.2)
Requirement already satisfied: toml<2,>=0.10.1 in c:\users\pavilion\appdata\local\programs\python\python313\lib\site-packages (from streamlit) (0.10.2)
Requirement already satisfied: typing-extensions<5,>=4.4.0 in c:\users\pavilion\appdata\local\programs\python\python313\lib\site-packages (from streamlit) (4.13.2)
Requirement already satisfied: watchdog<7,>=2.1.5 in c:\users\pavilion\appdata\local\programs\python\python313\lib\site-packages (from streamlit) (6.0.0)
Requirement already satisfied: gitpython!=3.1.19,<4,>=3.0.7 in c:\users\pavilion\appdata\local\programs\python\python313\lib\site-packages (from streamlit) (3.1.44)
Requirement already satisfied: pydeck<1,>=0.8.0b4 in c:\users\pavilion\appdata\local\programs\python\python313\lib\site-packages (from streamlit) (0.9.1)
Requirement already satisfied: tornado<7,>=6.0.3 in c:\users\pavilion\appdata\local\programs\python\python313\lib\site-packages (from streamlit) (6.5)
Requirement already satisfied: python-dateutil>=2.8.2 in c:\users\pavilion\appdata\local\programs\python\python313\lib\site-packages (from pandas) (2.9.0.post0)
Requirement already satisfied: pytz>=2020.1 in c:\users\pavilion\appdata\local\programs\python\python313\lib\site-packages (from pandas) (2025.2)
Requirement already satisfied: tzdata>=2022.7 in c:\users\pavilion\appdata\local\programs\python\python313\lib\site-packages (from pandas) (2025.2)
Requirement already satisfied: contourpy>=1.0.1 in c:\users\pavilion\appdata\local\programs\python\python313\lib\site-packages (from matplotlib) (1.3.2)
Requirement already satisfied: cycler>=0.10 in c:\users\pavilion\appdata\local\programs\python\python313\lib\site-packages (from matplotlib) (0.12.1)
Requirement already satisfied: fonttools>=4.22.0 in c:\users\pavilion\appdata\local\programs\python\python313\lib\site-packages (from matplotlib) (4.58.0)
Requirement already satisfied: kiwisolver>=1.3.1 in c:\users\pavilion\appdata\local\programs\python\python313\lib\site-packages (from matplotlib) (1.4.8)
Requirement already satisfied: pyparsing>=2.3.1 in c:\users\pavilion\appdata\local\programs\python\python313\lib\site-packages (from matplotlib) (3.2.3)
Requirement already satisfied: jinja2 in c:\users\pavilion\appdata\local\programs\python\python313\lib\site-packages (from altair<6,>=4.0->streamlit) (3.1.6)
Requirement already satisfied: jsonschema>=3.0 in c:\users\pavilion\appdata\local\programs\python\python313\lib\site-packages (from altair<6,>=4.0->streamlit) (4.23.0)
Requirement already satisfied: narwhals>=1.14.2 in c:\users\pavilion\appdata\local\programs\python\python313\lib\site-packages (from altair<6,>=4.0->streamlit) (1.39.1)
Requirement already satisfied: colorama in c:\users\pavilion\appdata\local\programs\python\python313\lib\site-packages (from click<9,>=7.0->streamlit) (0.4.6)
Requirement already satisfied: gitdb<5,>=4.0.1 in c:\users\pavilion\appdata\local\programs\python\python313\lib\site-packages (from gitpython!=3.1.19,<4,>=3.0.7->streamlit) (4.0.12)
Requirement already satisfied: six>=1.5 in c:\users\pavilion\appdata\local\programs\python\python313\lib\site-packages (from python-dateutil>=2.8.2->pandas) (1.17.0)
Requirement already satisfied: charset-normalizer<4,>=2 in c:\users\pavilion\appdata\local\programs\python\python313\lib\site-packages (from requests<3,>=2.27->streamlit) (3.4.2)
Requirement already satisfied: idna<4,>=2.5 in c:\users\pavilion\appdata\local\programs\python\python313\lib\site-packages (from requests<3,>=2.27->streamlit) (3.10)
Requirement already satisfied: urllib3<3,>=1.21.1 in c:\users\pavilion\appdata\local\programs\python\python313\lib\site-packages (from requests<3,>=2.27->streamlit) (2.4.0)
Requirement already satisfied: certifi>=2017.4.17 in c:\users\pavilion\appdata\local\programs\python\python313\lib\site-packages (from requests<3,>=2.27->streamlit) (2025.4.26)
Requirement already satisfied: smmap<6,>=3.0.1 in c:\users\pavilion\appdata\local\programs\python\python313\lib\site-packages (from gitdb<5,>=4.0.1->gitpython!=3.1.19,<4,>=3.0.7->streamlit) (5.0.2)
Requirement already satisfied: MarkupSafe>=2.0 in c:\users\pavilion\appdata\local\programs\python\python313\lib\site-packages (from jinja2->altair<6,>=4.0->streamlit) (3.0.2)
Requirement already satisfied: attrs>=22.2.0 in c:\users\pavilion\appdata\local\programs\python\python313\lib\site-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (25.3.0)
Requirement already satisfied: jsonschema-specifications>=2023.03.6 in c:\users\pavilion\appdata\local\programs\python\python313\lib\site-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (2025.4.1)
Requirement already satisfied: referencing>=0.28.4 in c:\users\pavilion\appdata\local\programs\python\python313\lib\site-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (0.36.2)
Requirement already satisfied: rpds-py>=0.7.1 in c:\users\pavilion\appdata\local\programs\python\python313\lib\site-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (0.25.0)

[notice] A new release of pip is available: 25.0.1 -> 25.1.1
[notice] To update, run: python.exe -m pip install --upgrade pip

C:\Users\PAVILION>streamlit run app.py
Usage: streamlit run [OPTIONS] TARGET [ARGS]...
Try 'streamlit run --help' for help.

Error: Invalid value: File does not exist: app.py

C:\Users\PAVILION>cd C:\Users\PAVILION\Downloads

C:\Users\PAVILION\Downloads>streamlit run app.py

  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8502
  Network URL: http://192.168.0.198:8502  
  ---

## وصف البيانات

البيانات المستخدمة تمثل جلسات لعب حقيقية وتتضمن:

- وقت بدء الجلسة  
- مدة اللعب (بالثواني)  
- عدد القفزات التي قام بها اللاعب  
- النقاط التي حصل عليها اللاعب  
- عدد العقبات التي تجاوزها اللاعب

---

## وصف التحليل والنماذج

- **تحليل استكشافي:** رسم العلاقات بين مدة اللعب وعدد القفزات، والعقبات والنقاط.  
- **تصنيف اللاعبين:** استخدام خوارزمية K-Means لتجميع اللاعبين حسب سلوكهم في اللعب.  
- **تصنيف أداء الجلسات:** استخدام نموذج Decision Tree لتوقع نجاح أو فشل الجلسة بناءً على الميزات.  
- **توصيات ذكية:** توليد توصيات مبنية على نتائج التحليل لتحسين تجربة اللعب.

---

## التحديات التي واجهتني

- توفير بيانات كافية ومعبرة عن سلوك اللاعبين.  
- تبسيط النماذج لضمان فهمها بسهولة والتفسير الصحيح للنتائج.  
- تصميم واجهة مستخدم تفاعلية واضحة وسهلة الاستخدام.

---

## الأفكار المستقبلية لتطوير المشروع

- تحليل تسلسل الأفعال لاكتشاف أنماط لعب متقدمة.  
- إضافة تقارير تفاعلية أكثر تفصيلاً مع إمكانيات تصفية وتحليل عميق.  
- دمج التطبيق مع اللعبة بشكل مباشر لجمع وتحليل البيانات في الوقت الحقيقي.

---

## كيفية المساهمة

هذا المشروع مفتوح للمساهمة لتحسين التحليل والواجهة أو إضافة مزايا جديدة.  
يرجى فتح Issues أو Pull Requests عبر GitHub.

---

## المراجع

- [scikit-learn Documentation](https://scikit-learn.org/stable/)  
- [Streamlit Documentation](https://docs.streamlit.io/)  
- [Google Colab](https://colab.research.google.com/)

---

## تواصل معي

لأي استفسار أو دعم:  
Email: s12218667@stu.najah.edu
GitHub: [shaymaa2004](https://github.com/shaymaa2004)

---

شكراً لاهتمامكم بمتابعة المشروع!
