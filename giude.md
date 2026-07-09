
# Xatran بی‌ربط — بهینه‌سازی و اپ یکپارچه برای sentiment-tweet-classification

## Context

پروژه در C:\Users\Masoudhp\Desktop\sentiment-tweet-classification یک پایپ‌لاین fine-tuning ترنسفورمر (BERT/RoBERTa/DistilBERT/ALBERT) برای طبقه‌بندی سه‌کلاسه احساسات ۴۵٬۶۱۵ توییت است، با ۹ کانفیگ YAML که دقیقاً ۵ محور خواسته‌شده در Project.pdf را پوشش می‌دهند (انتخاب مدل، LR/optimizer، پیش‌پردازش، max_length، fine-tuning strategy) و معیارهای accuracy/precision/recall/F1 روی dev را گزارش می‌کنند. معماری کد (src/) تمیز و منطقی است — split ثابت و stratified، dynamic padding، AMP، hardware monitor پس‌زمینه.

سه مشکل واقعی که باعث «چند ساعت انتظار، هدررفت منابع، و احتمال نتیجه غلط» می‌شود:

1. بازتوکنایز تکراری: TweetDataset.__getitem__ هر متن را در هر epoch و هر ارزیابی دوباره توکنایز می‌کند (۳ epoch × ۹ آزمایش = دهها هزار توکنایز اضافه) و DataLoader بدون num_workers/pin_memory است، یعنی توکنایز CPU با محاسبه GPU/CPU سری اجرا می‌شود.
2. بدون early stopping: حتی وقتی dev F1 چند epoch متوالی بهتر نشود، آموزش تا آخر ادامه می‌یابد (مثلاً bert_frozen_encoder با ۵ epoch).
3. بدون orchestration و بدون self-check: هر آزمایش با یک دستور جدا اجرا می‌شود، هیچ حالت میانی ذخیره/بازبینی نمی‌شود، و اگر کانفیگ اشتباه باشد این را فقط بعد از ساعت‌ها متوجه می‌شوید. همچنین هیچ ارزیابی روی test set در کل پروژه وجود ندارد — این یک نیاز صریح PDF است («گزارش نهایی فقط با test» ) که فعلاً جا افتاده.

کاربر خواسته: کد فقط از نظر سرعت/منابع بهینه شود (چون روی سیستم دیگری اجرا می‌شود، درایور GPU دست‌نخورده می‌ماند)، یک اپ Streamlit تک‌دستوری بسازیم که جریان اجرا را با دیاگرام state نشان دهد، لاگ زنده داشته باشد، و در پایان خودش نتیجه‌گیری کند. قبل از هر اجرای کامل (که می‌تواند ساعت‌ها طول بکشد)، یک «تست دود» خودکار روی زیرمجموعه کوچک داده و ۱ epoch اجرا شود تا خطاهای پایپ‌لاین/کانفیگ زود مشخص شوند.

## Approach

### ۱. بهینه‌سازی‌های ایمن در src/ (بدون تغییر منطق نتایج)

- `src/dataset.py`: TweetDataset را از توکنایز تنبل (per-__getitem__) به توکنایز حریصانه و دسته‌ای (batched tokenizer(texts, truncation=True, max_length=...) یک‌بار در __init__) تغییر بده. کالیبراسیون dynamic padding در collate دست‌نخورده می‌ماند. یک تابع کمکی make_dataloader(dataset, batch_size, shuffle, collate_fn, num_workers, pin_memory) اضافه کن.
- `src/config.py`: دو فیلد جدید با مقدار پیش‌فرض اضافه کن: early_stopping_patience: int | None = 2 و dataloader_num_workers: int = 2. چون این‌ها default دارند، فایل‌های YAML موجود بدون تغییر همچنان کار می‌کنند اما رفتار سریع‌تر می‌گیرند.
- `src/trainer.py`:
  - pin_memory=True وقتی device.type == "cuda"`، و `num_workers=config.dataloader_num_workers (با persistent_workers=True اگر num_workers>0).
  - Early stopping: اگر dev_f1 به مدت early_stopping_patience epoch پیاپی بهتر از best_dev_f1 نشود، حلقه را متوقف کن (checkpoint انتخابی همان best_state می‌ماند — نتیجه گزارش‌شده تغییر نمی‌کند، فقط زمان کمتر می‌شود).
  - GradScaler را به API جدید torch.amp.GradScaler(device_type, enabled=use_amp) با fallback به API قدیمی مهاجرت بده (رفع deprecation، بدون تغییر رفتار).
  - لاگ ساختاریافته: به‌جای فقط print`، از `logging استفاده کن که هم به کنسول و هم به results/<name>/train.log بنویسد.
  - بعد از هر epoch (و در نقاط کلیدی: شروع/پایان split، شروع/پایان هر آزمایش)، وضعیت را با src/run_state.py (جدید) به‌صورت atomic در results/run_state.json بنویس تا داشبورد آن را بخواند.
- `src/run_state.py` (جدید): کلاس سبک برای خواندن/نوشتن atomic یک JSON مشترک با اسکیمای: مرحله فعلی پایپ‌لاین (env_check/split/experiment/final_report/conclusions)، وضعیت هر آزمایش (pending|smoke_test|training|done|failed)، متریک epoch جاری، timestampها.

### ۲. رفع خلأ PDF: ارزیابی یک‌باره روی test set
- `scripts/final_report.py` (جدید): بعد از اتمام همه آزمایش‌ها، آزمایشی با بهترین dev F1 را از results/*/dev_metrics.json پیدا می‌کند، فقط همان یکی را روی splits.test ارزیابی می‌کند (با استفاده از evaluate_on_texts موجود در trainer.py)، و results/final_test_report.json + .md می‌نویسد. یک فایل قفل results/.test_set_evaluated.json (شامل نام آزمایش و timestamp) می‌سازد؛ اگر دوباره اجرا شود و آزمایش برنده عوض نشده باشد، از ارزیابی مجدد صرف‌نظر می‌کند و هشدار می‌دهد (برای رعایت قانون «test فقط یک‌بار» در PDF).

### ۳. تست دود خودکار (Smoke Test)

- در src/trainer.py یا یک ماژول جدید src/smoke_test.py: تابع run_smoke_test(config, train_texts, train_labels, dev_texts, dev_labels, n_samples=200) که با dataclasses.replace(config, epochs=1, name=f"{config.name}__smoke", output_dir="results/_smoke", monitor_hardware=False) و برش داده به n_samples نمونه، یک اجرای کامل ولی بسیار کوچک انجام می‌دهد. اگر خطا بدهد (OOM، کانفیگ نامعتبر، مشکل tokenizer)، آزمایش اصلی اصلاً شروع نمی‌شود و خطا در state ثبت می‌شود — این دقیقاً «self-check قبل از هدررفت ساعت‌ها» است.

### ۴. Orchestrator تک‌فایلی

- `scripts/run_all.py` (جدید): نقطه ورود منطقی پایپ‌لاین کامل:
  1. Env self-check: بررسی import شدن torch/transformers، `torch.cuda.is_available()`، تعداد CPU، فضای دیسک؛ نتیجه در state و کنسول.
  2. `prepare_splits` (idempotent، از src/data.py موجود).
  3. برای هر configs/*.yaml: اگر results/<name>/dev_metrics.json از قبل هست و --force نداده‌ایم → skip. وگرنه: smoke test → اگر پاس شد → run_training کامل → evaluate_on_texts روی dev (منطق فعلی run_experiment.py را این‌جا فراخوانی/بازاستفاده کن، کد را دوباره ننویس).
  4. scripts/compare_results.py منطق موجود را به‌عنوان تابع قابل فراخوانی (نه فقط __main__) دربیاور و اینجا صدا بزن.
  5. scripts/final_report.py (بخش ۲ بالا).
  6. `scripts/conclusions.py` (جدید): با مقایسه هر ردیف comparison_table.csv نسبت به bert_baseline`، یک `results/findings.md تولید می‌کند: بهترین مدل، اثر LR/optimizer/max_length/freeze/پیش‌پردازش (دقیقاً محورهای بخش «تحلیل نتایج» و «جمع‌بندی» PDF).
  - run_all.py با آرگومان‌های ساده (--force, --configs, --skip-smoke) از CLI هم قابل اجراست، هم از داخل اپ Streamlit به‌عنوان subprocess صدا زده می‌شود.

### ۵. اپ Streamlit تک‌دستوری (`app.py`، جدید)

- دکمه «شروع اجرای کامل» که `scripts/run_all.py` را در subprocess پس‌زمینه اجرا می‌کند (خروجی هم به فایل لاگ می‌رود، هم استریم زنده در UI).
- دیاگرام state: به‌جای وابستگی به باینری سیستمی Graphviz (که ممکن است روی سیستم مقصد نصب نباشد)، یک دیاگرام سبک HTML/CSS (جعبه‌های رنگی: خاکستری=pending، آبی چشمک‌زن=running، سبز=done، قرمز=failed) با st.markdown(unsafe_allow_html=True) رسم می‌شود؛ گره‌ها: Env Check → Split → هر آزمایش (Smoke→Train→Eval) → Compare → Final Test Report → Conclusions.
- لاگ زنده: تیل آخرین N خط از train.log آزمایش جاری (خواندن دوره‌ای فایل + st.rerun هر ۲ ثانیه با st.session_state timer، بدون نیاز به کامپوننت خارجی).
- نمودارها (Plotly، سبک و بدون باینری سیستمی): بار-چارت مقایسه accuracy/precision/recall/F1 بین آزمایش‌ها، نمودار CPU/RAM/GPU از hardware_log.csv (بازنویسی منطق plot_hardware.py با Plotly تعاملی به‌جای matplotlib استاتیک)، Gantt کوچک از مدت‌زمان هر آزمایش.
- نتیجه‌گیری خودکار: نمایش محتوای results/findings.md و results/final_test_report.md در تب پایانی.
- `run.bat` (جدید، ریشه پروژه): streamlit run app.py — تنها دستوری که کاربر لازم است اجرا کند.

### ۶. وابستگی‌های جدید

requirements.txt: افزودن streamlit>=1.35 و plotly>=5.20 (توجیه در README: برای داشبورد تک‌دستوری، بدون وابستگی به باینری سیستمی خارجی مثل Graphviz).

### فایل‌های دست‌نخورده (فقط بازاستفاده می‌شوند، نه تغییر معنایی)

src/data.py, src/metrics.py, src/model.py, src/preprocessing.py, src/error_analysis.py, scripts/prepare_splits.py, scripts/verify_gpu.py, همه فایل‌های configs/*.yaml (فقط به‌خاطر default جدید در دیتاکلاس رفتار سریع‌تر می‌گیرند، بدون نیاز به ادیت دستی).

## Verification
- torch/transformers/streamlit روی این ماشین نصب نیستند و طبق تصمیم کاربر قرار است روی سیستم دیگری اجرا شود؛ بنابراین یک اجرای واقعی fine-tuning اینجا قابل تأیید نیست.
- کاری که اینجا قابل انجام و لازم است:
  - نصب سبک pyyaml, scikit-learn, pandas, streamlit, plotly در یک venv موقت برای تست غیر-torch بخش‌ها.
  - اجرای python scripts/prepare_splits.py واقعی (چون فقط sklearn لازم دارد) برای تأیید split.
  - اجرای بخش env-check و state-writer run_all.py به‌صورتی که وقتی torch/transformers غایب است، پیام واضح بدهد و در state ثبت کند (نه crash خام) — این مسیر خطا را عمداً تست می‌کنیم چون دقیقاً حالت این ماشین است.
  - اجرای streamlit run app.py روی این ماشین برای دیدن رندر داشبورد با state ساختگی/خالی (بدون نیاز به آموزش واقعی).
  - python -m py_compile روی همه فایل‌های جدید/تغییریافته برای اطمینان از نبود خطای نحوی، به‌علاوه بازبینی دستی منطق early stopping و قفل test-set.
- در پایان به کاربر توضیح می‌دهم که تست end-to-end واقعی fine-tuning را باید روی سیستم مقصد (با torch نصب‌شده) خودش با run.bat انجام دهد و چک‌لیست کوتاهی برای آن سیستم می‌دهم (نصب requirements.txt + requirements-torch.txt`، اجرای `run.bat).