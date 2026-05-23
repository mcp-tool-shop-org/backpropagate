<p align="center">
  <a href="README.ja.md">日本語</a> | <a href="README.zh.md">中文</a> | <a href="README.es.md">Español</a> | <a href="README.fr.md">Français</a> | <a href="README.md">English</a> | <a href="README.it.md">Italiano</a> | <a href="README.pt-BR.md">Português (BR)</a>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/mcp-tool-shop-org/brand/main/logos/backpropagate/readme.png" alt="Backpropagate" width="400">
</p>

<p align="center">
  <a href="https://github.com/mcp-tool-shop-org/backpropagate/actions/workflows/ci.yml"><img src="https://github.com/mcp-tool-shop-org/backpropagate/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://pypi.org/project/backpropagate/"><img src="https://img.shields.io/pypi/v/backpropagate" alt="PyPI"></a>
  <a href="https://codecov.io/gh/mcp-tool-shop-org/backpropagate"><img src="https://img.shields.io/codecov/c/github/mcp-tool-shop-org/backpropagate" alt="Coverage"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT License"></a>
  <a href="https://mcp-tool-shop-org.github.io/backpropagate/"><img src="https://img.shields.io/badge/Landing_Page-live-blue" alt="Landing Page"></a>
</p>

"हेडलेस एलएलएम (LLM) को केवल तीन पंक्तियों में फाइन-ट्यून करें। इसमें स्मार्ट डिफ़ॉल्ट सेटिंग्स, वीआरएएम (VRAM) को ध्यान में रखते हुए बैच का आकार निर्धारित करने की सुविधा, मल्टी-रन एसएलएओ (SLAO), और ओलामा (Ollama) के लिए वन-क्लिक जीजीयूएफ (GGUF) एक्सपोर्ट जैसी विशेषताएं हैं।"

*SLAO का अर्थ है "सिंगल लोरा निरंतर शिक्षण, असममित विलय के माध्यम से" - यह एक विलय तकनीक है जो विस्तारित फाइन-ट्यूनिंग प्रक्रियाओं के दौरान "विनाशकारी भूलने" को रोकने में मदद करती है ([पेपर](https://arxiv.org/abs/2512.23017)।*

*सिर्फ तीन पंक्तियों के कोड का उपयोग करके बड़े भाषा मॉडल (LLMs) को प्रशिक्षित करें। फिर, इसे एक और पंक्ति के कोड से ओलामा (Ollama) में एक्सपोर्ट करें।*

## शुरुआत कैसे करें।

```bash
pip install backpropagate[standard]
```

```python
from backpropagate import Trainer

trainer = Trainer("Qwen/Qwen2.5-7B-Instruct")
trainer.train("examples/quickstart.jsonl", steps=10)
trainer.export("gguf", quantization="q4_k_m")  # Ready for Ollama
```

इस रिपॉजिटरी में एक छोटा सा `examples/quickstart.jsonl` फ़ाइल (जिसमें 5 शेयरजीपीटी-फॉर्मेट के उदाहरण हैं) शामिल है, ताकि ऊपर दिया गया कोड एक नए इंस्टॉलेशन पर पूरी तरह से काम कर सके। अपने स्वयं के प्रशिक्षण के लिए, नीचे दिए गए "[डेटासेट फॉर्मेट](#dataset-format)" अनुभाग को देखें।

### कोडिंग की आवश्यकता न होने वाला तरीका: वेब इंटरफ़ेस।

क्या आप पाइथन के इंटरैक्टिव शेल (REPL) की बजाय एक ग्राफिकल यूजर इंटरफेस (UI) पसंद करते हैं? तो, उसी अतिरिक्त पैकेज को स्थापित करें और फिर इसे चलाएं:

```bash
pip install backpropagate[standard]
backprop ui --port 7862
```

"रिफ्लेक्स (रेडिक्स यूआई) इंटरफेस आपको एक JSONL फ़ाइल चुनने, एक मॉडल का चयन करने, प्रशिक्षण करने और निर्यात करने की सुविधा देता है - इसके लिए पाइथन की आवश्यकता नहीं है। यह इंटरफेस स्थानीय रूप से काम करता है; सार्वजनिक इंटरनेट पर उपयोग के लिए, नीचे दिए गए "[वेब यूआई](#web-ui)" अनुभाग में `--share` और `--auth` सुरक्षा प्रोटोकॉल, साथ ही समर्थित टनल विकल्पों (क्लाउडफ्लेयर टनल, एनग्रोक) के बारे में जानकारी दी गई है।"

## डेटासेट का प्रारूप।

आपकी जेएसओएनएल (JSONL) प्रशिक्षण फ़ाइल में प्रत्येक पंक्ति में एक उदाहरण होना चाहिए। सबसे सरल प्रारूप "शेयरजीपीटी" (ShareGPT) चैट है:

```jsonl
{"conversations": [{"from": "human", "value": "What is Python?"}, {"from": "gpt", "value": "A programming language."}]}
{"conversations": [{"from": "human", "value": "Explain recursion."}, {"from": "gpt", "value": "A function that calls itself."}]}
```

अल्पाका (निर्देश/आउटपुट), ओपनएआई चैट (संदेश), और साधारण टेक्स्ट फॉर्मेट भी समर्थित हैं। एक शुरुआती बिंदु के लिए, `examples/quickstart.jsonl` फ़ाइल देखें, जिसे आप कॉपी कर सकते हैं।

## बैकप्रोपगेशन क्यों किया जाता है?

| समस्या। | समाधान। |
|---------|----------|
| बारीक समायोजन एक जटिल प्रक्रिया है। | 3 पंक्तियाँ: लोड (लोड), ट्रेन (ट्रेन), सेव (सेव)। |
| विंडोज एक दुःस्वप्न जैसा है। | विंडोज के लिए उत्कृष्ट समर्थन। |
| वीआरएएम (VRAM) का प्रबंधन एक जटिल कार्य है। | स्वचालित बैच आकार निर्धारण, जीपीयू (GPU) की निगरानी। |
| मॉडल का निर्यात एक जटिल प्रक्रिया है। | एक क्लिक में GGUF और Ollama का पंजीकरण। |
| लंबे समय तक लगातार चलने से याददाश्त कमजोर हो सकती है। | एकाधिक बार दोहराए जाने वाले एसएलएओ प्रशिक्षण। |

## मुख्य विशेषताएं।

- **डिज़ाइन द्वारा हेडलेस:** सीआई/सीडी पाइपलाइनों, स्वचालित प्रक्रियाओं और प्रोग्रामेटिक निष्पादन के लिए निर्मित।
- **स्मार्ट डिफ़ॉल्ट सेटिंग्स:** आपके हार्डवेयर और डेटासेट के आधार पर, स्वचालित रूप से इष्टतम हाइपरपैरामीटर कॉन्फ़िगर करता है।
- **मल्टी-रन एसएलएओ प्रशिक्षण:** लंबे समय तक चलने वाले प्रशिक्षण के दौरान विनाशकारी भूलने से बचाने के लिए उन्नत प्रशिक्षण रणनीतियाँ।
- **विंडोज के लिए उत्कृष्ट समर्थन:** विंडोज वातावरण में परीक्षण किया गया और अनुकूलित, जो सामान्य पायटॉर्च/CUDA समस्याओं से बचाता है।
- **आसान निर्यात:** एक क्लिक में जीजीयूएफ प्रारूप में निर्यात करें और ओलामा के साथ स्वचालित रूप से पंजीकृत करें।
- **मॉड्यूलर आर्किटेक्चर:** केवल उन निर्भरताओं को स्थापित करें जिनकी आपको आवश्यकता है (उदाहरण के लिए, `[unsloth]`, `[ui]`, `[export]`)।

## स्थापना।

```bash
pip install backpropagate              # Core only (minimal)
pip install backpropagate[unsloth]     # + Unsloth 2x faster training
pip install backpropagate[ui]          # + Reflex (Radix UI) web interface
pip install backpropagate[standard]    # unsloth + ui (recommended)
pip install backpropagate[full]        # Everything
```

| अतिरिक्त। | विवरण। | निर्भरताएँ। |
|-------|-------------|--------------|
| `unsloth` | 2 गुना तेज़ प्रशिक्षण, 50% कम वीआरएएम (VRAM) की आवश्यकता। | अनस्लोथ। |
| `ui` | रिफ्लेक्स (रेडिक्स यूआई) वेब इंटरफेस। | "reflex" का संस्करण 0.9.2 या उससे अधिक होना चाहिए, और "fastapi" का संस्करण 0.115 या उससे अधिक होना चाहिए। |
| `validation` | पायडैंटिक का कॉन्फ़िगरेशन सत्यापन। | pydantic, pydantic-सेटिंग्स |
| `export` | ओलामा के लिए जीजीयूएफ प्रारूप में डेटा का निर्यात। | llama-cpp-python |
| `monitoring` | वैंडबी (WandB) और सिस्टम मॉनिटरिंग (संस्करण 1.1.0 में ट्रेनर में स्वचालित रूप से एकीकृत)। | wandb, psutil |
| `logging` | संरचित लॉगिंग। | स्ट्रक्टलॉग (structlog) एक ऐसा उपकरण है। |
| `security` | JWT प्रमाणीकरण (ऑथेंटिकेशन) और टोकन निर्माण। | PyJWT, क्रिप्टोग्राफी। |
| `production` | अनस्लोथ + यूआई (यूजर इंटरफेस) + सत्यापन + लॉगिंग + सुरक्षा। | (गुच्छा) |

**आवश्यकताएं:** पायथन 3.10 या उससे ऊपर का संस्करण, CUDA जीपीयू (8 जीबी या उससे अधिक वीआरएएम), पायटॉर्च 2.0 या उससे ऊपर का संस्करण।

### प्लेटफ़ॉर्म की आवश्यकताएं।

बैकप्रोपगेट रनटाइम संबंधी समस्याओं (मल्टीप्रोसेसिंग, RTX 40/50 पर एक्सफॉर्मर्स, विंडोज पर डेटालोडर वर्कर्स) को संभालता है। यह इंस्टॉलेशन के समय होने वाली प्लेटफ़ॉर्म संबंधी समस्याओं को **नहीं** संभालता है - पहले उन्हें ठीक करें:

- **CUDA टूलकिट संस्करण।** PyTorch को CUDA संस्करण के अनुसार जारी किया जाता है - गलत व्हील चुनने से केवल CPU वाला torch इंस्टॉल हो सकता है। अपने ड्राइवर के लिए सटीक `pip install torch ...` कमांड के लिए <https://pytorch.org/get-started/locally/> पर दिए गए टूल का उपयोग करें। अपने ड्राइवर/CUDA संस्करण को देखने के लिए `nvidia-smi` चलाएं।
- **विंडोज।** `[export]` एक्सट्रा के लिए Visual Studio Build Tools (C++) और CMake की आवश्यकता होती है (llama-cpp-python स्रोत कोड से बनाया जाता है)। अब `bitsandbytes` व्हील विंडोज के लिए मूल रूप से उपलब्ध है (>= 0.43); `bitsandbytes-windows` का उल्लेख करने वाले पुराने गाइड पुराने हैं।
- **macOS।** GPU प्रशिक्षण समर्थित **नहीं** है - कोई CUDA नहीं। आप Ollama के माध्यम से निर्यात किए गए GGUF पर *अनुमान* चलाने के लिए बैकप्रोपगेट इंस्टॉल कर सकते हैं, लेकिन `trainer.train()` `DEP_GPU_NOT_AVAILABLE` त्रुटि उत्पन्न करता है। प्रशिक्षण के लिए CUDA मशीन का उपयोग करें।
- **लिनक्स।** अधिकांश वितरण डिफ़ॉल्ट रूप से काम करते हैं। यदि आप PyPI बाइनरी रिलीज़ का उपयोग कर रहे हैं, तो ध्यान दें कि लिनक्स बिल्ड केवल CPU वाला torch का उपयोग करता है (GitHub की 2 GB रिलीज़-एसेट सीमा से नीचे रहने के लिए); पहले pytorch.org से संबंधित CUDA व्हील इंस्टॉल करें।

लंबे समय तक चलने वाली इंस्टॉलेशन संबंधी समस्याओं के लिए, [समस्या निवारण मार्गदर्शिका पृष्ठ](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting/) देखें।

## कॉन्फ़िगरेशन

सभी सेटिंग्स को `BACKPROPAGATE_` उपसर्ग (जैसे, `BACKPROPAGATE_LOG_LEVEL=debug`) का उपयोग करके पर्यावरण चर के साथ ओवरराइड किया जा सकता है। प्रोजेक्ट रूट में एक `.env` फ़ाइल स्वचालित रूप से लोड हो जाती है जब `[validation]` एक्सट्रा इंस्टॉल किया जाता है।

सामान्य सेटिंग्स (हर चीज़ के लिए [पूरे पर्यावरण चर संदर्भ](https://mcp-tool-shop-org.github.io/backpropagate/handbook/env-vars/)):

| चर | डिफ़ॉल्ट | टिप्पणियाँ |
|----------|---------|-------|
| `BACKPROPAGATE_LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `BACKPROPAGATE_LOG_JSON` | ऑटो | JSON लॉग (`true`) या कंसोल लॉग (`false`) को फ़ोर्स करें |
| `BACKPROPAGATE_LOG_FILE` | अनिर्धारित | लॉग को किस स्थान पर सहेजा जाए |
| `BACKPROPAGATE_DEFER_FEATURE_DETECTION` | अनिर्धारित | सबसे तेज़ CLI कोल्ड स्टार्ट के लिए स्टार्टअप पर वैकल्पिक-निर्भरता का पता लगाना छोड़ दें |
| `BACKPROPAGATE_SECURITY__REQUIRE_AUTH_FOR_SHARE` | `true` | जब `true` होता है, तो `--auth` के बिना `backprop ui --share` को अस्वीकार कर देता है |
| `BACKPROPAGATE_UI__OUTPUT_DIR` | `~/.backpropagate/ui-outputs` | सभी UI फ़ाइल सिस्टम राइट्स के लिए सैंडबॉक्स बेस; डिनाइलिस्ट-सत्यापित |
| `BACKPROPAGATE_MODEL__NAME` | `Qwen/Qwen2.5-7B-Instruct` | डिफ़ॉल्ट मॉडल |
| `BACKPROPAGATE_TRAINING__LEARNING_RATE` | `2e-4` | लर्निंग रेट |
| `BACKPROPAGATE_LORA__R` | `16` | LoRA रैंक |

नेस्टेड कुंजियों के लिए विभाजक के रूप में डबल अंडरस्कोर का उपयोग किया जाता है (Pydantic `env_nested_delimiter` कन्वेंशन)।

## उपयोग

### बुनियादी प्रशिक्षण

```python
from backpropagate import Trainer

trainer = Trainer("Qwen/Qwen2.5-7B-Instruct")
trainer.train("my_data.jsonl", steps=100)
trainer.save("./my-model")
trainer.export("gguf", quantization="q4_k_m")
```

`Qwen/Qwen2.5-7B-Instruct` डिफ़ॉल्ट विकल्प है — `Trainer()` फ़ंक्शन को बिना किसी मॉडल तर्क के कॉल करने पर यही मान निर्धारित होता है (देखें [`config.py`](backpropagate/config.py) में `ModelConfig.name`)। पुराने उदाहरणों में पहले से क्वांटाइज किया गया `unsloth/Qwen2.5-7B-Instruct-bnb-4bit` उपयोग किया गया था; हमने बेहतर विश्वसनीयता के लिए डिफ़ॉल्ट को आधिकारिक Qwen मॉडल भार में बदल दिया ([CHANGELOG v1.1.0](CHANGELOG.md#110---2026-05-21))। दोनों मॉडल काम करते हैं।

### मल्टी-रन SLAO प्रशिक्षण

```python
from backpropagate import Trainer

trainer = Trainer("Qwen/Qwen2.5-7B-Instruct")

result = trainer.multi_run(
    dataset="HuggingFaceH4/ultrachat_200k",
    num_runs=5,
    steps_per_run=100,
    samples_per_run=1000,
    merge_mode="slao",  # Single LoRA Continual Learning via Asymmetric Merging
)
```

SLAO (सिंगल LoRA कंटीन्यूअल लर्निंग वाया एसिमेट्रिक मर्जिंग) [मर्ज बिफोर फॉरगेट](https://arxiv.org/abs/2512.23017) पेपर को लागू करता है: QR अपघटन के माध्यम से ऑर्थोगोनल A-मैट्रिक्स इनिशियलाइज़ेशन, एसिमेट्रिक A/B हैंडलिंग, और समय-जागरूक `λ(i) = 1/√i` स्केलिंग। CLI फ़्लैग `--samples` है (अंतर्निहित फ़ील्ड `samples_per_run` है)।

### Ollama में निर्यात करें

```python
# Export to GGUF
result = trainer.export("gguf", quantization="q4_k_m")

# Register with Ollama separately
from backpropagate import register_with_ollama
register_with_ollama(result.path, "my-finetuned-model")
# ollama run my-finetuned-model
```

### CLI

```bash
backprop train --data my_data.jsonl --model Qwen/Qwen2.5-7B-Instruct --steps 100
backprop multi-run --data my_data.jsonl --runs 5 --steps 100
backprop export ./output/lora --format gguf --quantization q4_k_m --ollama --ollama-name my-model
backprop ui --port 7862
backprop info
backprop list-runs                              # v1.1.0: query past training runs
backprop show-run <run-id>                      # v1.1.0: detail view
backprop resume <run-id>                        # v1.1.0: resume a crashed multi-run
backprop push ./output/lora --repo me/my-model  # v1.1.0: push adapter to HF Hub
```

प्रत्येक सबकमांड और फ़्लैग के लिए [CLI संदर्भ](https://mcp-tool-shop-org.github.io/backpropagate/handbook/cli-reference/) देखें, या `backprop <subcommand> --help` चलाएं।

### चेकपॉइंट से फिर से शुरू करें (v1.1.0)

एक 5-चरण वाला मल्टी-रन जो 4वें चरण पर क्रैश हो जाता है, अब उसे ठीक किया जा सकता है। प्रत्येक मल्टी-रन सत्र अपनी `run_id` को `run_history.json` और डिस्क पर मौजूद चेकपॉइंट मैनिफेस्ट दोनों में लिखता है, इसलिए जहां से आपने छोड़ा था, वहीं से शुरू करने के लिए केवल एक कमांड की आवश्यकता होती है:

```bash
backprop resume <run-id>                       # picks up the in-progress session
backprop multi-run --data ... --resume <run-id> # explicit form
backprop train --data ... --resume <run-id>    # single-run resume (continues run_id)
```

`backprop multi-run` का डिफ़ॉल्ट व्यवहार (बिना `--resume`) स्वचालित रूप से उसी आउटपुट डायरेक्टरी में चल रहे सत्र का पता लगाता है और उसे जारी रखता है। `resume_from="off"` (पाइथन एपीआई) पास करें या `--resume` को छोड़ दें और एक नए आउटपुट डायरेक्टरी में शुरू करें ताकि एक नया सत्र शुरू हो सके।

जब कोई मल्टी-रन फिर से शुरू होता है, तो उस `run_id` के लिए नवीनतम चेकपॉइंट मॉडल में लोड किया जाता है, `slao/` डायरेक्टरी से SLAO विलय स्थिति को पुनर्स्थापित किया जाता है, और रन लूप `last_completed_run + 1` से जारी रहता है। इतिहास प्रविष्टि की `status` वापस `running` में बदल जाती है, इसलिए `backprop list-runs --status running` लाइव सत्र दिखाता है।

### प्रयोग ट्रैकिंग (v1.1.0)

`Trainer` स्वचालित रूप से स्थापित प्रयोग ट्रैकर (`wandb`, `tensorboard`, `mlflow`) का पता लगाता है और उन्हें अंतर्निहित `transformers.TrainingArguments` में एकीकृत करता है। डिफ़ॉल्ट `report_to="auto"` उन सभी को चुनता है जिन्हें इम्पोर्ट किया जा सकता है:

```bash
pip install backpropagate[monitoring]  # installs wandb + psutil
wandb login                            # one-time
backprop train --data my_data.jsonl    # W&B run gets the same run_id prefix as the on-disk history
```

स्पष्ट रूप से बाहर निकलने के लिए `Trainer(report_to=["wandb"])`, `Trainer(report_to=["tensorboard"])`, या `Trainer(report_to="none")` का उपयोग करें। MLflow के लिए, `pip install mlflow` जोड़ें; TensorBoard के लिए, `pip install tensorboard` जोड़ें। W&B रन का नाम `backprop-<run_id_prefix>` है, ताकि एक ऑपरेटर W&B, हमारे लॉग और `run_history.json` में एक ही पहचानकर्ता का उपयोग करके खोज कर सके।

### प्रशिक्षण इतिहास

प्रत्येक `backprop train` और `backprop multi-run` कमांड `<output>/run_history.json` में एक पंक्ति रिकॉर्ड करता है, जिसमें `run_id`, मॉडल, डेटासेट, हाइपरपैरामीटर, स्थिति, अंतिम हानि, हानि इतिहास और (मल्टी-रन के लिए) SLAO विलय समयरेखा शामिल होती है। हाल के रन की सूची देखें:

```bash
backprop list-runs                         # most recent 20 runs, all statuses
backprop list-runs --status failed         # filter
backprop list-runs --json --limit 100      # machine-readable
backprop show-run abcd1234                 # detail view (partial run_id ok)
```

रन इतिहास प्रक्रियाओं में भी बना रहता है - वेब UI में `Runs` टैब एक अलग, इन-मेमोरी दृश्य है; डिस्क पर मौजूद इतिहास `list-runs` / `show-run` / `resume` के लिए स्रोत है।

### वेब UI

स्थानीय रूप से Reflex इंटरफ़ेस लॉन्च करें:

```bash
backprop ui --port 7862
```

सार्वजनिक इंटरनेट URL दिखाने के लिए, आपको `--share` को `--auth` के साथ जोड़ना होगा:

```bash
backprop ui --share --auth alice:hunter2
```

`backprop ui --share` कमांड, `--auth` विकल्प के बिना चलाने पर, कोड `1` के साथ और एक संरचित त्रुटि संदेश `[RUNTIME_UI_AUTH_NOT_ENFORCED]` प्रदर्शित करता है। इसका कारण यह है कि `--share` एक सार्वजनिक यूआरएल प्रकाशित करता है, जिसे इंटरनेट पर कोई भी व्यक्ति एक्सेस कर सकता है। प्रमाणीकरण (ऑथ) के बिना, इसका मतलब है कि कोई भी व्यक्ति आपके प्रशिक्षण प्रक्रिया को नियंत्रित कर सकता है। यदि आप प्रमाणीकरण सेट नहीं करना चाहते हैं, तो एसएसएच पोर्ट-फॉरवर्डिंग का उपयोग करें: `ssh -L 7860:localhost:7860 <होस्ट>` और फिर `http://localhost:7860` को स्थानीय रूप से खोलें। पूर्ण खतरे के मॉडल के लिए, [handbook/security.md](site/src/content/docs/handbook/security.md) देखें।

UI से किए गए फ़ाइल सिस्टम लेखन को एक ही डायरेक्टरी तक सीमित कर दिया गया है:

- डिफ़ॉल्ट: `~/.backpropagate/ui-outputs`
- ओवरराइड: `BACKPROPAGATE_UI__OUTPUT_DIR=/path/you/own`
- ओवरराइड को **ब्लैकलिस्ट-सत्यापित** किया गया है - सिस्टम/क्रेडेंशियल पथ (`/etc`, `/var`, `~/.ssh`, `~/.aws`, `C:\Windows\System32`, आदि) को `[UI_OUTPUT_DIR_FORBIDDEN]` के साथ अस्वीकार कर दिया जाता है।

## विंडोज समर्थन

Backpropagate को डिफ़ॉल्ट रूप से विंडोज पर काम करने के लिए डिज़ाइन किया गया है:

- मल्टीप्रोसेसिंग क्रैश से बचने के लिए प्री-टोकनाइजेशन
- RTX 40/50 श्रृंखला के लिए स्वचालित xformers अक्षम
- सुरक्षित डेटा लोडर सेटिंग्स
- RTX 5080 (16GB VRAM) पर परीक्षण किया गया

## मॉडल प्रीसेट

| प्रीसेट | VRAM | गति | गुणवत्ता |
|--------|------|-------|---------|
| Qwen 2.5 7B | ~12GB | मध्यम | सर्वोत्तम |
| Qwen 2.5 3B | ~8GB | तेज़ | अच्छा |
| Llama 3.2 3B | ~8GB | तेज़ | अच्छा |
| Llama 3.2 1B | ~6GB | सबसे तेज़ | बुनियादी |
| Mistral 7B | ~12GB | मध्यम | अच्छा |

## आर्किटेक्चर

```
backpropagate/
├── trainer.py           # Core Trainer class
├── multi_run.py         # Multi-run SLAO training
├── slao.py              # SLAO LoRA merging algorithm
├── datasets.py          # Dataset loading, filtering & curriculum
├── export.py            # GGUF/Ollama export
├── config.py            # Pydantic settings + training presets
├── gpu_safety.py        # GPU monitoring & safety
├── cli.py               # CLI entry point (backprop command)
├── checkpoints.py       # Checkpoint management
├── exceptions.py        # Structured error hierarchy
├── feature_flags.py     # Optional feature detection
├── security.py          # Path traversal & torch security
├── logging_config.py    # Structured logging setup
├── ui_theme.py          # Radix theme tokens + CSS (Reflex era)
├── ui_state.py          # rx.State subclasses
├── ui_app/              # Reflex web interface (Radix UI)
│   ├── app.py           #   rx.App entry point
│   ├── chrome.py        #   Header / LeftNav / SideRail / Footer
│   ├── pages/           #   Train / Multi-Run / Export / Dataset
│   └── components/      #   Bp* primitives (status pill, sparkline, event log…)
└── ui_security.py       # Rate limiting, CSRF, file validation (framework-agnostic)
```

v1.0 Gradio कार्यान्वयन (`ui_gradio_legacy.py` + `theme_gradio_legacy.py`) को v1.1.x तक संदर्भ के रूप में रखा गया था और v1.2.0 में हटा दिया गया।

## समस्या निवारण

सबसे आम शुरुआती विफलताओं का एक संक्षिप्त विवरण। पूर्ण रिवर्स इंडेक्स [समस्या निवारण पुस्तिका पृष्ठ](https://mcp-tool-shop-org.github.io/backpropagate/handbook/troubleshooting/) पर उपलब्ध है; नीचे दिया गया प्रत्येक कोड [त्रुटि कोड](https://mcp-tool-shop-org.github.io/backpropagate/handbook/error-codes/) पर प्रलेखित है।

| लक्षण | कोड | समाधान |
|---------|------|-----|
| प्रशिक्षण के दौरान GPU की मेमोरी समाप्त हो जाती है। | `RUNTIME_GPU_OOM` | OOM ऑटो-रिकवरी (B-002) बैच आकार को स्वचालित रूप से 3 बार तक आधा कर देता है। इसे बंद करने के लिए: `Trainer(oom_recovery=False)`। बैच आकार को कम करने के लिए: `--batch-size 1`। |
| HF हब 401 / "मॉडल नहीं मिला" त्रुटि देता है। | `DEP_MODEL_LOAD_FAILED` | `huggingface-cli login` चलाएं और फिर से प्रयास करें। टाइपिंग की गलतियों के लिए, <https://huggingface.co/models> से सटीक आईडी कॉपी करें। |
| मॉडल के नाम में टाइपिंग की गलती। | `INPUT_VALIDATION_FAILED` या `DEP_MODEL_LOAD_FAILED` | <https://huggingface.co/models> पर `org/name` पहचानकर्ता की जांच करें। |
| `register_with_ollama` कनेक्शन अस्वीकृत। | `DEP_OLLAMA_REGISTRATION_FAILED` | डेमॉन शुरू करें: `ollama serve`। <https://ollama.com> से इंस्टॉल करें। पुनः प्रयास करने योग्य। |
| चेकपॉइंट सहेजते समय डिस्क भर गई। | `STATE_CHECKPOINT_INVALID` | क्रैश होने पर एटॉमिक राइट्स `.partial` नामक एक डायरेक्टरी बनाते हैं - इसे हटाना सुरक्षित है। पिछला अच्छा चेकपॉइंट बरकरार है। |
| GPU के अत्यधिक गर्म होने के कारण प्रशिक्षण रुक गया/रद्द कर दिया गया। | `RUNTIME_GPU_TEMPERATURE_CRITICAL` | B-003 मॉनिटर NVML तापमान सीमा पर रुक जाता है; GPU के ठंडा होने पर यह स्वचालित रूप से फिर से शुरू हो जाता है। वायु प्रवाह में सुधार करें या निरंतर लोड को कम करें। |
| `backprop ui --share` अस्वीकृत। | `INPUT_AUTH_REQUIRED` | `--auth user:password` पास करें, या `BACKPROPAGATE_SECURITY__REQUIRE_AUTH_FOR_SHARE=false` सेट करें (चेतावनी)। |
| मल्टी-रन "वैलिडेशन ओवरलैप"। | `CONFIG_INVALID` (स्टेज A बैकएंड B-001) | `--samples` को प्रशिक्षण पूल आकार से कम करें, डेटासेट बढ़ाएं, या वैलिडेशन को अक्षम करें। |
| GGUF एक्सपोर्ट पहली बार में विफल रहा। | `RUNTIME_GGUF_EXPORT_FAILED` | `pip install backpropagate[export]`। विंडोज पर, आपको Visual C++ Build Tools + CMake की भी आवश्यकता है। |

## बग की रिपोर्ट करना

जब कुछ विफल होता है, तो Backpropagate स्टार्टअप पर `run_started run_id=<uuid>` लाइन प्रिंट करता है और उसी आईडी को चेकपॉइंट मैनिफेस्ट, SLAO मर्ज इतिहास और संरचित लॉग लाइनों से जोड़ता है। किसी भी बग रिपोर्ट में `run_id` शामिल करें - यह एक रखरखावकर्ता को उस विशिष्ट रन के लिए प्रत्येक लॉग लाइन, प्रत्येक चेकपॉइंट और प्रत्येक मर्ज को सहसंबंधित करने की अनुमति देता है।

एक अच्छी बग रिपोर्ट में शामिल हैं:

1. **`run_id`** — स्टार्टअप पर प्रिंट किया गया UUID (यह `TrainingRun.run_id` और `RunResult.run_id` के रूप में भी उपलब्ध है)।
2. **त्रुटि कोड** — stderr में `[CODE_NAME]: message` लाइन वह है जिसे आपको खोजना चाहिए; कैटलॉग के लिए [त्रुटि कोड](https://mcp-tool-shop-org.github.io/backpropagate/handbook/error-codes/) देखें।
3. **रेडैक्टेड कमांड लाइन।** गैर-विस्तृत मोड में stderr स्वचालित रूप से रेडैक्ट किया जाता है (Bearer टोकन, `sk-*`, `hf_*`, AWS कुंजियाँ, `password=`/`token=`/`api_key=` जोड़े हटा दिए जाते हैं) - इसे पेस्ट करना सुरक्षित है। पूर्ण, बिना रेडैक्ट किए गए ट्रेसबैक के लिए, `--verbose` के साथ फिर से चलाएं, लेकिन पोस्ट करने से पहले इसकी समीक्षा करें।
4. **Python / PyTorch संस्करण, GPU मॉडल, OS।** `backprop info` यह सब एक साथ प्रिंट करता है।

## गोपनीयता

सभी प्रशिक्षण आपके GPU पर स्थानीय रूप से होता है। Backpropagate केवल HuggingFace से मॉडल डाउनलोड करने के लिए नेटवर्क अनुरोध करता है (जो आप शुरू करते हैं)। कोई टेलीमेट्री नहीं, कोई क्लाउड निर्भरता नहीं।

## स्कोरकार्ड

| श्रेणी | स्कोर | टिप्पणियाँ |
|----------|-------|-------|
| A. सुरक्षा | 6/8 | SECURITY.md, ट्रस्ट मॉडल, कोई गुप्त/टेलीमेट्री नहीं, safe_path()। MCP आइटम छोड़े गए। |
| B. त्रुटि प्रबंधन | 5/7 | संरचित त्रुटि स्वरूप (`कोड`/`संदेश`/`संकेत`/`कारण`/`पुन: प्रयास करने योग्य`) त्रुटि कोड रजिस्ट्री के माध्यम से; CLI (कमांड लाइन इंटरफेस) के लिए 0/1/2/3 एग्जिट कोड; `--verbose` के बिना कोई कच्चा स्टैक ट्रेस नहीं; `run_id` सहसंबंध; संपादित stderr; `--share` + `--auth` गेटिंग। MCP/डेस्कटॉप/वीएस कोड को छोड़ दिया गया। |
| सी. ऑपरेटर दस्तावेज़ | 4/7 | README, CHANGELOG, LICENSE, --help। लॉगिंग/MCP/जटिल चीज़ों को छोड़ दिया गया। |
| डी. शिपिंग स्वच्छता | 6/9 | verify.sh, संस्करण=टैग, CI में 5 स्कैनर, डिपेंडabot, python_requires, स्वच्छ बिल्ड। |
| ई. पहचान | 4/4 | लोगो, अनुवाद, लैंडिंग पृष्ठ, मेटाडेटा। |
| **Total** | **25/31** | 14 आइटम छोड़े गए, जिसके कारण बताए गए हैं · `shipcheck audit` 100% पास करता है · ऑडिट की तारीख: 2026-05-21 (बी-पंक्ति को स्टेज बी + स्टेज ए CLI एग्जिट-कोड कार्य के बाद फिर से वर्गीकृत किया गया)। |

डिजाइन इतिहास और प्रत्येक पंक्ति आइटम का क्या अर्थ है: [ROADMAP.md](ROADMAP.md) देखें — सभी सप्ताह 1–4 के आइटम v1.1.0 में जारी किए गए हैं।

## लाइसेंस

MIT — विवरण के लिए [LICENSE](LICENSE) देखें।

---

<p align="center">
  Built by <a href="https://mcp-tool-shop.github.io/">MCP Tool Shop</a>
</p>
