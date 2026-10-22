[English README](README.md)

# Arabic Dialect Toolkit

نظام تجريبي لتحديد اللهجة العربية، ومعالجة المصرية إلى الفصحى عند ثقة كافية.
المشروع يفصل بين مهمتين: **Dialect Identification** و **Egyptian → MSA Normalization**.

## الحالة الحالية

النسخة الأولى تشمل تنظيفًا محافظًا للنص العربي، وتقسيمًا stratified ثابتًا، وbaseline بـ
character TF-IDF + LinearSVC، وpipeline، وواجهة FastAPI، واختبارات للوحدات الأساسية.

البيانات نفسها لا تدخل Git. أضف ملفاتك المرخّصة تحت `data/raw/` بعد قراءة شروط الاستخدام.
تعليمات المصادر والـ adapters موجودة في [`data/README.md`](data/README.md).

## التشغيل على PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
pytest
```

## تجهيز الداتا

بعد وضع الملفات تحت `data/raw/` شغّل:

```powershell
python -m src.data.prepare
```

بيعمل: تنضيف IADD (حذف `general` والمكرر والمتضارب والجمل الطويلة جداً والكلاسات الصغيرة)، وإزالة
التكرار والتسريب بين train/valid/test في داتا المصري→فصحى (ويقرأ `data/gold/egy_msa_gold.csv` لو موجود)،
ويكتب `data/processed/*.csv` وتقرير `reports/data_audit.md`. كلاس MSA مش موجود في IADD؛
الخيار `--msa-from-parallel N` بيضيفه من الجانب الفصيح للداتا المتوازية، لكن اختلاف المجال بيخلّي
تمييزه سهل جداً، فاعتبر نتيجته مبالغاً فيها.

للتدريب على الناتج:

```powershell
python -m src.dialect_id.train_eval --input data/processed/dialect.csv --output-dir reports/baseline
```

ولو عندك CSV خاص بيك بعمودين `text` و`label`:

```powershell
python -m src.dialect_id.baseline --input data/raw/train.csv --output models/dialect_baseline.joblib
```

ولتقسيم البيانات وتقييمها وإخراج التقرير:

```powershell
python -m src.dialect_id.train_eval --input data/raw/train.csv --output-dir reports/baseline
```

سيُنتج الأمر `metrics.json` و`confusion_matrix.csv` ونسخة النموذج. الصيغ المدعومة حاليًا
هي CSV وJSONL، والأعمدة المطلوبة `text` و`label`. الـ loader يحتفظ بـ `text_raw`، يوحّد
أسماء labels الشائعة، ويرفض النصوص المتطابقة بعد التنظيف إذا كانت labels مختلفة.

لمقارنة LinearSVC مع Naive Bayes على نفس التقسيم:

```powershell
python -m src.dialect_id.compare --input data/raw/train.csv --output-dir reports/comparison
```

سيُنتج `comparison.json` مع Macro-F1 وباقي المقاييس لكل نموذج.

الـ confidence الإنتاجية لا ينبغي أن تعتمد على margin خام من LinearSVC. استخدم
`src.dialect_id.calibrated` للتدريب مع probability calibration، ثم استخدم
`src.eval.threshold.select_threshold` على validation set لاختيار threshold المصرية.
لا تستخدم test set في اختيار threshold.

لتفعيل المعايرة أثناء التدريب:

```powershell
python -m src.dialect_id.train_eval --input data/raw/train.csv `
  --output-dir reports/calibrated --calibrated
```

سيُحفظ threshold المختار في `threshold.json`.

لحساب Identity baseline على test:

```powershell
python -m src.eval.translation
```

ينتج `reports/identity_baseline.json` ويحسب BLEU وchrF. عند تثبيت `sacrebleu` يستخدم
التنفيذ القياسي؛ وإلا يستخدم fallback محليًا واضحًا في حقل `backend`. أرقام الـ fallback **غير قابلة للمقارنة**
مع sacrebleu، فقارن النموذج بـ Identity من نفس الـ backend فقط.

لتدريب أول نموذج seq2seq على GPU:

```powershell
python -m src.normalizer.train_seq2seq --model-name google/mt5-small `
  --train data/processed/egy_msa/train.jsonl `
  --valid data/processed/egy_msa/valid.jsonl `
  --output-dir models/egy_msa_mt5 --lr 1e-4
```

(`fp16` مقفول افتراضيًا لأن mT5 ممكن يطلّع NaN؛ فعّله بـ `--fp16` على مسؤوليتك.)

التدريب طويل نسبيًا؛ شغّله على Colab أو Kaggle إذا لم تتوفر GPU محلية.

لتقييم النموذج على test جنب Identity بنفس المقاييس (يكتب `reports/mt5_test.json` والتوقعات):

```powershell
python -m src.normalizer.evaluate --model-dir models/egy_msa_mt5
```

أو شغّل `notebooks/colab_egy_msa.ipynb` على Colab لكل الخطوات.

لتجربة النموذج بعد التدريب:

```powershell
python -m src.normalizer.predict --model-dir models/egy_msa_mt5 `
  --text "انا مش عارف اروح فين"
```

لتشغيل الـ API:

```powershell
uvicorn app.main:app --reload
```

## API

- `POST /identify` — يحدد اللهجة.
- `POST /normalize` — يستخدم نموذج mT5 لو فولدر `models/egy_msa_mt5` موجود (يتحمّل مرة واحدة عند التشغيل)، وإلا يستخدم placeholder بسيط. الحقل `backend` في الرد بيوضّح أيهما اشتغل.
- `POST /pipeline` — يرجع التحويل فقط عند مصرية وثقة فوق threshold.
- `GET /health` — يوضح هل نموذج اللهجات موجود وأي normalizer شغال.

المسارات الافتراضية: `reports/dialect_baseline/dialect_baseline.joblib` و`models/egy_msa_mt5`، ويمكن تغييرها بمتغيرات البيئة `DIALECT_MODEL_PATH` و`NORMALIZER_MODEL_DIR`.

## خطة البيانات والتقييم

ابدأ بـ IADD بعد التحقق من الترخيص، ثم أضف NADI/MADAR وفق شروط الوصول وإعادة التوزيع.
احتفظ بـ gold set مصري مكتوب ومراجع يدويًا خارج كل مراحل التدريب، وقيّم Macro-F1، مصفوفة
الالتباس، وcross-dataset evaluation. بالنسبة للتحويل، قارن دائمًا بـ identity baseline وأضف
chrF/BLEU وتقييمًا بشريًا قبل الادعاء بجودة النموذج.

