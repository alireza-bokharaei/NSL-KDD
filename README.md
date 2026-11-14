# Reproducibility package — preliminary NSL-KDD benchmark

**English** | [فارسی](#نسخه-فارسی)

---

## English

This package accompanies a draft manuscript, not a peer-reviewed or accepted publication. It does not evaluate an LLM penetration-testing agent or a live IDS/IPS.

### Data and attribution
NSL-KDD `KDDTrain+.TXT` and `KDDTest+.TXT` were obtained from https://github.com/jmnwong/NSL-KDD-Dataset at commit `9d544d0eb9b87d7e2f43ff65733bdb644631d12f`. They are included as `data/KDDTrain.txt` and `data/KDDTest.txt`. The original provider says the data are no longer hosted there and permits redistribution with citation: https://www.unb.ca/cic/datasets/nsl.html.

Cite: Tavallaee, M., Bagheri, E., Lu, W. & Ghorbani, A. A. (2009). A Detailed Analysis of the KDD CUP 99 Data Set. IEEE CISDA, 1–6. https://doi.org/10.1109/CISDA.2009.5356528.

SHA-256:
- `KDDTrain.txt`: `1b86d2f957b33082081bba410fe129b475efebcc13c9014c3f447c8271aadf95`
- `KDDTest.txt`: `fa46b0935342616aa83b7c2578db355b6a7aaabbc492248172c7a1e8b7ab8f84`

### Main experiment
Python 3.13; numpy 2.3; pandas 2.2; scikit-learn 1.6, Install `requirements.txt`, then run `python run_multiseed.py` from the folder. Results overwrite `results_multiseed.json`.

Splits are stratified 80/20 from KDDTrain+ with seeds 7, 42, and 101. Each fitted model is evaluated on its corresponding holdout and on the same fixed KDDTest+, so both evaluations use an identical model for each seed. All preprocessing is fit on the corresponding training subset. There is no tuning on KDDTest+; this multi-seed analysis followed an earlier exploratory single-seed analysis and was not preregistered.

An exact 41-feature merge against the entire KDDTrain+ identifies 664 overlapping KDDTest+ rows; they are excluded only for the sensitivity analysis. This conservative criterion uses all training records, including holdout records, not just each training subset. Model training is unchanged. Full test and filtered test have different compositions.

Results are descriptive: the shared test set, overlapping splits, dataset age, and dependence within records limit statistical inference. Three seeds do not establish uncertainty intervals or external generalization.

`run_experiment.py` and `results.json` preserve the prior full-training single-seed exploratory experiment; they are not the source of the updated manuscript tables. The two scripts can take several minutes. Float values may differ slightly across machines. The data, scripts and recorded results accompany the research manuscript.

---

## نسخه فارسی

# بستهٔ بازتولیدپذیری — بنچمارک مقدماتی NSL-KDD

این بسته همراه یک پیش‌نویس مقاله است و نه یک انتشار داوری‌شده یا پذیرفته‌شده. این بسته یک عامل تست نفوذ مبتنی بر LLM یا یک IDS/IPS زنده را ارزیابی نمی‌کند.

### داده‌ها و ارجاع‌دهی
فایل‌های `KDDTrain+.TXT` و `KDDTest+.TXT` مربوط به NSL-KDD از مخزن https://github.com/jmnwong/NSL-KDD-Dataset در کامیت `9d544d0eb9b87d7e2f43ff65733bdb644631d12f` دریافت شده‌اند و به‌صورت `data/KDDTrain.txt` و `data/KDDTest.txt` در این بسته قرار دارند. تأمین‌کنندهٔ اصلی اعلام کرده است که داده‌ها دیگر در آن آدرس میزبانی نمی‌شوند و بازتوزیع آن‌ها با ذکر منبع مجاز است: https://www.unb.ca/cic/datasets/nsl.html

ارجاع:
Tavallaee, M., Bagheri, E., Lu, W. & Ghorbani, A. A. (2009). A Detailed Analysis of the KDD CUP 99 Data Set. IEEE CISDA, 1–6. https://doi.org/10.1109/CISDA.2009.5356528

SHA-256:
- `KDDTrain.txt`: `1b86d2f957b33082081bba410fe129b475efebcc13c9014c3f447c8271aadf95`
- `KDDTest.txt`: `fa46b0935342616aa83b7c2578db355b6a7aaabbc492248172c7a1e8b7ab8f84`

### آزمایش اصلی
Python 3.13؛ numpy 2.3؛ pandas 2.2؛ scikit-learn 1.6. ابتدا `requirements.txt` را نصب کنید، سپس از داخل پوشه دستور `python run_multiseed.py` را اجرا کنید. نتایج فایل `results_multiseed.json` را بازنویسی می‌کنند.

تقسیم‌بندی‌ها به‌صورت طبقه‌بندی‌شده (stratified) با نسبت ۸۰/۲۰ از KDDTrain+ و با بذرهای ۷، ۴۲ و ۱۰۱ انجام شده‌اند. هر مدلِ برازش‌شده روی holdout متناظر خود و همچنین روی همان KDDTest+ ثابت ارزیابی می‌شود؛ بنابراین هر دو ارزیابی برای هر بذر از یک مدل یکسان استفاده می‌کنند. تمام پیش‌پردازش‌ها روی زیرمجموعهٔ آموزش متناظر برازش شده‌اند. هیچ تنظیم hyperparameter روی KDDTest+ انجام نشده است؛ این تحلیل چندبذری پس از یک تحلیل اکتشافی قبلی تک‌بذری انجام شده و از پیش ثبت (preregistered) نشده بود.

یک ادغام دقیق بر اساس هر ۴۱ ویژگی با کل KDDTrain+، تعداد ۶۶۴ سطر هم‌پوشان در KDDTest+ را شناسایی می‌کند؛ این سطرها فقط برای تحلیل حساسیت حذف می‌شوند. این معیار محافظه‌کارانه از تمام رکوردهای آموزش — از جمله رکوردهای holdout، نه فقط زیرمجموعهٔ آموزش هر بذر — استفاده می‌کند. آموزش مدل تغییری نمی‌کند. مجموعهٔ آزمون کامل و مجموعهٔ آزمون فیلترشده ترکیب متفاوتی دارند.

نتایج توصیفی هستند: مجموعهٔ آزمون مشترک، تقسیم‌بندی‌های هم‌پوشان، قدمت داده‌ها و وابستگی میان رکوردها امکان استنباط آماری را محدود می‌کنند. سه بذر نمی‌توانند بازه‌های عدم‌قطعیت یا تعمیم‌پذیری بیرونی را اثبات کنند.

فایل‌های `run_experiment.py` و `results.json` آزمایش اکتشافی قبلی با آموزش کامل و تک‌بذر را حفظ می‌کنند و منبع جداول به‌روزشدهٔ مقاله نیستند. اجرای این دو اسکریپت ممکن است چند دقیقه طول بکشد. مقادیر اعشاری ممکن است بین ماشین‌های مختلف اندکی متفاوت باشند. داده‌ها، اسکریپت‌ها و نتایج ثبت‌شده همراه دست‌نوشتهٔ پژوهشی ارائه می‌شوند.

---

[English](#reproducibility-package--preliminary-nsl-kdd-benchmark) | **فارسی**
