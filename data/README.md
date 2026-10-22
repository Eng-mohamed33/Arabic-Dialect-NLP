# Local data instructions

الملفات الأصلية لا تُرفع إلى Git. ضعها في `data/raw/` بعد مراجعة شروط كل مصدر.

## Dialect identification

- **IADD**: نزّل `IADD.json` من [المستودع الرسمي](https://github.com/JihadZa/IADD). الحقول الأساسية هي `Sentence` و`Region` و`Country` و`DataSource`. الـ adapter يستبعد `general` ويحتفظ بـ Iraq كـ `Iraqi`؛ لا تخلطه تلقائيًا مع Gulf.
- **NADI**: التسجيل مطلوب من [المستودع الرسمي](https://github.com/UBC-NLP/nadi). لا تضع ملفات التنزيل أو بيانات تويتر في repository.
- **QADI**: المستودع يحتوي IDs للتغريدات، وليس النصوص نفسها؛ يلزم hydration وفق الشروط الموضحة في README.
- **MADAR**: اطلب الوصول من [صفحة المشروع](https://sites.google.com/nyu.edu/madar/).

## Egyptian → MSA

- **EGY_MSA_Translation**: ضع ملفات `EGY-parallel_{train,valid,test}_*.txt` و`msa_parallel_{train,valid,test}_*.txt` (6 ملفات) تحت `data/raw/egy_msa_translation/`. استخدم `load_parallel_text` وتأكد من تطابق عدد الأسطر.
- **EgyptianBench**: استخدمه كتقييم خارجي فقط إذا درّبت على EGY_MSA_Translation، لأن بطاقة البيانات تذكر وجود تداخل مصدر. لا تخلط train وtest.

## مثال

```python
from src.data.load import load_iadd, load_parallel_text

iadd = load_iadd("data/raw/IADD.json")
parallel = load_parallel_text(
    "data/raw/egy_msa_translation/EGY-parallel_valid_13358.txt",
    "data/raw/egy_msa_translation/msa_parallel_valid_45049.txt",
)
```


## بعد التجهيز

`python -m src.data.prepare` بيكتب `data/processed/dialect.csv` و`parallel_{train,valid,test}.csv`
وتقرير `reports/data_audit.md`. الـ Gold set بتاعك بيتحط في `data/gold/egy_msa_gold.csv` (شوف `data/gold/README.md`).
