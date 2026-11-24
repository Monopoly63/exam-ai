# مذاكرة عملي AI — HTML

الملف الأساسي: **`index.html`** — صفحة عربية واحدة، بدون أي CDN أو مكتبات خارجية، فيها:

- خريطة خطوات السبورة الخمسة (Import Library → Import Data → Pre-processing → Train → Test / Prediction / Figure out).
- أكواد **Optimization** الخمسة: SGD، SGD + Momentum/Decay، AdaGrad، RMSprop، Adam.
- أكواد **Regularization**: Early Stopping، Dropout، Batch Normalization، مع بطاقة إضافية عن L2 و `C`.
- أكواد **SVM / SVC / SVR**: البايبلاين، Kernels، `C` / `gamma` / `epsilon`، GridSearchCV ونسخة آمنة بـ Pipeline.
- قالب جاهز، ١٢ سؤال تدريب مع شرح، Checklist محفوظة بالمتصفح، بحث، وضع حفظ، وضع طباعة، ونسخ الكود بضغطة.

## التشغيل

أسهل شيء: افتح `index.html` بالمتصفح مباشرة. أو شغّل سيرفر محلي:

```bash
python3 -m http.server 4173 --bind 0.0.0.0 --directory study
```

بعدين افتح `http://localhost:4173/`.

## الملفات

| الملف | الفائدة |
| --- | --- |
| `index.html` | المراجعة التفاعلية الأساسية |
| `exam_templates.py` | نفس القوالب بصيغة Python للنسخ السريع — انسخ قسم واحد، لا تشغّل الملف كامل |
| `sources/*.py` | نسخ كاملة من أكواد المحاضرات الأصلية المستخرجة من ملفات ZIP |
| `build_page.py` | مولّد الصفحة؛ عدّله ثم `python3 build_page.py` إذا بدك تغيّر المحتوى |

## المصادر

من الـ ZIP الموجودة بجذر الريبو: `3rd_Lecture_Codes_AI_Practical.zip`، `5th_Lecture_Codes.zip`، `AI_Lectture_4.zip`. خريطة الملفات كاملة موجودة بآخر قسم بالصفحة.
