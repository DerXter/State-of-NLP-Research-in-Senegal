# Opportunities and Challenges of Natural Language Processing for Low-Resource __Senegalese__ Languages in Social Science Research 🇸🇳 cf [Survey Paper](https://www.preprints.org/manuscript/202601.1124)

This work presents a comprehensive survey of natural language processing (NLP) for six major Senegalese national languages: `Wolof`, `Pulaar`, `Sereer`, `Mandinka`, `Soninké`, and `Joola`. We provide an overview of the current state of research across key NLP tasks and highlight persistent challenges related to data scarcity, orthographic variation, and linguistic diversity. In addition, we introduce this centralized and openly accessible repository that compiles existing datasets, benchmarks, and tools available for these languages. The repository is designed as a __living resource to be periodically expanded through community contributions__. Our objective is to map existing efforts, identify critical research gaps, and encourage the development of sustainable, inclusive NLP research for Senegal’s national languages.

If you are interested in the state of the art of NLP research in **African languages more broadly**, you can take a look at this [comprehensive, two-decade survey of AfricaNLP research (2005–2025), analyzing publications, authors, affiliations, supporters, NLP topics, and tasks](https://arxiv.org/abs/2509.25477v3). The major linguistic and sociopolitical challenges that hinder the development of NLP technologies for African languages are discussed in this [Afrocentric NLP paper](https://arxiv.org/abs/2203.08351).

## Taxonomy
We drew inspiration from the taxonomy proposed in the [Awesome AI Papers repository](https://github.com/aimerou/awesome-ai-papers/) to propose this one. We chose subjective limits in terms of `number of citations` and use a set of **icons** to highlight which paper meets which criteria.

⭐ `Important Paper` : more than 50 citations and state of the art results.

⏫ `Trend` : 1 to 50 citations, innovative paper with growing adoption.

📰 `Important Article` : decisive work that was not accompanied by a research paper.

An added 🔐 icons means that no open version (`locked access`) for this article was found. The 🌍 icon means the paper has been plubished by **non-senegalese** African authors and the icon 🌐 indicates papers published by foreign authors (**outside of Africa**). We consider the paper to be `regional` 🌍 or `international` 🌐 if: 
* the first author paper is an African or a foreigner; 
* the project leading to the paper was launched by Africans or foreigners.
    > e.g. The [MasakhaPOS](https://aclanthology.org/2023.acl-long.609/) paper has a senegalese as a first author but the overall research project has been launched by [Masakhane](https://www.masakhane.io/).

Finally, since Senegal is a French-speaking country, some of the articles were written in French. We thus added the 🇫🇷 icon to highlight those papers.

> A paper may also appear in multiple sections if it covers various domains, tasks, and/or modalities.

----

## Datasets 🔃
- The [Online Wolof Data](https://github.com/WolofProcessing/online_wolof_data) repository tracks and centralizes all `openly accessible datasets` as well as potential `data sources` on the `Wolof` language.
- We extended this Wolof repository to the `other 05 national languages` in the [Datasets](Datasets.md) file.

## NLP Tools
|Name|Covered tasks |Languages supported|
|----------|-----------|----------------|
|[Wolof keyboards](https://github.com/srheal/Wol_Keyboards) | Keyboards for MacOS, Android and Apple mobile | Wolof    |
|[Stanza](https://stanfordnlp.github.io/stanza/) ([Qi et al., 2020](https://arxiv.org/abs/2003.07082))    | Part-Of-Speech (POS) and Morphological features tagging dependency parsing| Wolof    |
|[MorphScore](https://github.com/catherinearnett/morphscore) ([Arnett et al., 2025](https://arxiv.org/abs/2507.06378))    |   Morphological alignment evaluation      |   Wolof     |
|[Wolof](https://github.com/abdouaziz/wolof)| *Fill_mask* (Masked Language Modeling)|   Wolof             |
|[Common Voice](https://commonvoice.mozilla.org/wo) ([Ardila et al., 2019](https://arxiv.org/abs/1912.06670)) , [DVoice](https://www.dvoice.africa/) ([Allak et al., 2021](https://www.semanticscholar.org/paper/Dialectal-Voice-%3A-An-Open-Source-Voice-Dataset-and-Allak-Mohamed/3f70a1d61d324f3ca199e9089a1344021f74332e))    |   Speech Data Collection      |   Wolof    |
|[AfroLID](https://github.com/UBC-NLP/afrolid), [GlotLID](https://github.com/cisnlp/GlotLID), [AfroScope](https://github.com/UBC-NLP/AfroScope)  |   Language Identification      |   Wolof     |


## Publications
For some articles, we were also unable to find open versions highlighted by the 🔐 icon.
> If you need access to some of the locked papers, feel free to reach out at `derguenembaye[at]esp[dot]sn`.

### Digraphs
- ⏫ 05/2020: [Digraph of Senegal s local languages: issues, challenges and prospects of their transliteration](https://arxiv.org/abs/2005.02325)
- ⏫🇫🇷 05/2020: [Digraphie des langues ouest africaines : Latin2Ajami : un algorithme de translitteration automatique](https://arxiv.org/abs/2005.02827)
- ⏫ 01/2025: [The Best of Both Worlds: Exploring Wolofal in the Context of NLP](https://aclanthology.org/2025.abjadnlp-1.1/)
- ⏫🌐🇫🇷 06/2025: [Réhabiliter l’écriture Ajami : un levier technologique pour l’alphabétisation en Afrique](https://aclanthology.org/2025.jeptalnrecital-recital.14/)

### Parsing & Tokenization
- ⏫ 05/2012: [A Morphological Analyzer For Wolof Using Finite-State Techniques](https://aclanthology.org/L12-1324/)
- ⏫🔐 06/2013: [Handling Wolof clitics in LFG](https://www.degruyterbrill.com/document/doi/10.1075/la.206.04dio)
- ⏫ 08/2013: [ParGramBank: The ParGram Parallel Treebank](https://aclanthology.org/P13-1054/)
- ⏫ 05/2014: [Pruning the Search Space of the Wolof LFG Grammar Using a Probabilistic and a Constraint Grammar Parser](https://aclanthology.org/L14-1497/)
- ⏫ 08/2014: [LFG parse disambiguation for Wolof](https://www.researchgate.net/publication/269556176_LFG_parse_disambiguation_for_Wolof)
- ⏫ 11/2017: [Finite-State Tokenization for a Deep Wolof LFG Grammar](https://bells.uib.no/index.php/bells/article/view/1340)
- ⏫ 08/2019: [Developing Universal Dependencies for Wolof](https://aclanthology.org/W19-8003/)
- ⏫ 05/2020: [Implementation and Evaluation of an LFG-based Parser for Wolof](https://aclanthology.org/2020.lrec-1.631/)
- ⏫ 12/2020: [From LFG To UD: A Combined Approach](https://aclanthology.org/2020.udw-1.7/)
- ⏫ 08/2021: [Multilingual Dependency Parsing for Low-Resource African Languages: Case Studies on Bambara, Wolof, and Yoruba](https://aclanthology.org/2021.iwpt-1.9/)
- ⏫🌐 07/2025: [Evaluating Morphological Alignment of Tokenizers in 70 Languages](https://arxiv.org/abs/2507.06378)

### Language Identification
- ⏫🌍 10/2022: [AfroLID: A Neural Language Identification Tool for African Languages](https://arxiv.org/abs/2210.11744)
- ⭐🌐 12/2023: [GlotLID: Language Identification for Low-Resource Languages](https://aclanthology.org/2023.findings-emnlp.410/)
- ⏫🌍 01/2026: [AfroScope: A Framework for Studying the Linguistic Landscape of Africa](https://arxiv.org/abs/2601.13346v1) 🔃

### Linguistic Similarity, Embeddings & Cross-Lingual Transfer
- ⭐🌐 11/2020: [Extending Multilingual BERT to Low-Resource Languages](https://aclanthology.org/2020.findings-emnlp.240/) 🔃
- ⭐🌐 07/2025: [Analyzing the Effect of Linguistic Similarity on Cross-Lingual Transfer: Tasks and Experimental Setups Matter](https://aclanthology.org/2025.findings-acl.454/)
- ⏫🌐 01/2026: [Can Embedding Similarity Predict Cross-Lingual Transfer? A Systematic Study on African Languages](https://arxiv.org/abs/2601.03168)
- ⏫ 02/2026: [Cross-lingual Matryoshka Representation Learning across Speech and Text](https://arxiv.org/abs/2602.19991v1) 🔃
- ⏫🌐 03/2026: [Omnilingual SONAR: Cross-Lingual and Cross-Modal Sentence Embeddings Bridging Massively Multilingual Text and Speech](https://ai.meta.com/research/publications/omnilingual-sonar-cross-lingual-and-cross-modal-sentence-embeddings-bridging-massively-multilingual-text-and-speech/) 🔃

### Token Classification

#### POS Tagging
- ⏫ 05/2010: [Design and Development of Part-of-Speech-Tagging Resources for Wolof (Niger-Congo, spoken in Senegal)](https://aclanthology.org/L10-1228/)
- ⏫🌍 07/2023: [MasakhaPOS: Part-of-Speech Tagging for Typologically Diverse African languages](https://aclanthology.org/2023.acl-long.609/)

#### Named Entity Recognition
- ⏫🌍 03/2021: [MasakhaNER: Named Entity Recognition for African Languages](https://aclanthology.org/2021.tacl-1.66/)
- ⏫🌍 12/2022: [MasakhaNER 2.0: Africa-centric Transfer Learning for Named Entity Recognition](https://aclanthology.org/2022.emnlp-main.298/)

### Text Classification

#### Opinion Mining / Sentiment Analysis
- ⏫ 06/2018: [A Novel Term Weighting Scheme Model](https://dl.acm.org/doi/10.1145/3233347.3233374)
- ⏫🔐 03/2019: [FWLSA-score: French and Wolof Lexicon-based for Sentiment Analysis](https://ieeexplore.ieee.org/document/8714667)
- ⏫🔐 12/2019: [Improved Bilingual Sentiment Analysis Lexicon Using Word-level Trigram](https://ieeexplore.ieee.org/document/9064223)
- ⏫🔐 07/2020: [SenOpinion: a new lexicon for opinion tagging in Senegalese news comments](https://ieeexplore.ieee.org/abstract/document/9140887)
- ⏫🇫🇷 06/2022: [COMFO : Corpus Multilingue pour la Fouille d’Opinions (COMFO: Multilingual Corpus for Opinion Mining)](https://aclanthology.org/2022.jeptalnrecital-taln.29/)
    > The English version of this article is available on [Springer Nature Link](https://link.springer.com/chapter/10.1007/978-3-031-19907-3_2). 
- ⏫🔐 06/2023: [Markov Model for French-Wolof Text Analysis](https://ieeexplore.ieee.org/document/10242779)
- ⏫ 08/2024: [A lexicon-based sentiment analysis approach using a graph structure for modeling relationships between opinion words in French and Wolof corpora](https://dl.acm.org/doi/10.1145/3676581.3676594)
- ⏫ 10/2025: [Sentiment Analysis on the Young People's Perception About the Mobile](https://link.springer.com/chapter/10.1007/978-981-96-9709-0_14)

#### Hate Speech Detection
- ⏫🌍 06/2023: [Towards hate speech detection in low-resource languages: Comparing ASR to acoustic word embeddings on Wolof and Swahili](https://arxiv.org/abs/2306.00410)
- ⏫ 06/2025: [Annotated tweet corpus of mixed Wolof-French for detecting obnoxious messages](https://www.sciencedirect.com/science/article/pii/S235234092500232X)
- ⏫🔐 07/2025: [Comparative Study of Machine Learning Models for the Detection of Abusive Messages: Case of Wolof-French Codes Mixing Data](https://link.springer.com/chapter/10.1007/978-3-031-86493-3_20)
- ⏫ 09/2025: [AbuseBERT-WoFr: refined BERT model for detecting abusive messages on tweets mixing Wolof-French codes](https://hal.science/hal-05249237v1/file/Proceedings%2520of%2520Digital%2520Avenues%2520for%2520Low-Resource%2520Languages%2520of%2520Sub-Saharan%2520Africa%2520%2528DASSA%25E2%2580%25992025%2529.pdf)
    > `Page 10` of the _Proceedings of Digital Avenues for Low-Resource Languages of Sub-Saharan Africa (DASSA’2025)_.

#### Intent Classification
- ⏫🌍 02/2025: [INJONGO: A Multicultural Intent Detection and Slot-filling Dataset for 16 African Languages](https://arxiv.org/abs/2502.09814)
    > Note: Intent classification is generally performed with slot-filling (which is token classification) as a [joint task](https://arxiv.org/abs/2101.08091) to maximize performance in both tasks simultaneously.
- ⏫ 09/2025: [WolBanking77: Wolof Banking Speech Intent Classification Dataset](https://arxiv.org/abs/2509.19271)

### Lexicons and Spell Checking
- ⏫🌐🇫🇷 01/2015: [DILAF : des dictionnaires africains en ligne et une méthodologie](https://hal.science/hal-01107550)
- ⏫🇫🇷 03/2016: [Dictionnaires wolof en ligne: État de l'art et perspectives](https://hal.science/hal-01294544)
- ⏫🇫🇷 03/2016: [Production et mise en ligne d’un dictionnaire électronique du wolof](https://talnarchives.atala.org/ateliers/2016/TALAf/10.pdf)
- ⏫🇫🇷 03/2016: [iBaatukaay : un projet de base lexicale multilingue contributive sur le web à structure pivot pour les langues africaines notamment sénégalaises](https://hal.science/hal-02054921)
- ⏫🇫🇷 07/2016: [Correction orthographique pour la langue wolof: état de l'art et perspectives](http://hal.science/hal-02054917)
- ⏫🇫🇷 09/2018: [Manipulation de dictionnaires d'origines diverses pour des langues peu dotées: la méthodologie iBaatukaay](https://hal.science/hal-01992863v1)
- ⏫ 05/2023: [Automatic Spell Checker and Correction for Under-represented Spoken Languages: Case Study on Wolof](https://arxiv.org/abs/2305.12694)
- ⏫ 05/2024: [Advancing language diversity and inclusion: Towards a neural network-based spell checker and correction for wolof](https://aclanthology.org/2024.rail-1.16/)
- ⏫ 07/2024: [Beqi: Revitalize the senegalese wolof language with a robust spelling corrector](https://arxiv.org/pdf/2305.08518)
- 📰🇫🇷 09/2025: [SenTermino - Banque Terminologique Scientifique du Sénégal](https://sentermino.com/fr/)

### Machine Translation
- ⏫ 03/2020: [Using LSTM Networks to Translate French to Senegalese Local Languages: Wolof as a Case Study](https://arxiv.org/abs/2004.13840)
- ⏫ 05/2020: [Sencorpus: A french-wolof parallel corpus](https://aclanthology.org/2020.lrec-1.341/)
- ⏫ 08/2020: [Building word representations for wolof using neural networks](https://link.springer.com/chapter/10.1007/978-3-030-51051-0_20)
- ⭐🌐 10/2020: [Beyond English-Centric Multilingual Machine Translation](https://arxiv.org/abs/2010.11125)
- ⏫ 03/2022: [SenTekki: Online Platform and Restful Web Service for Translation Between Wolof and French](https://link.springer.com/chapter/10.1007/978-3-031-23116-2_25)
- ⏫ 06/2022: [Low-resource neural machine translation: Benchmarking state-of-the-art transformer for Wolof<->French](https://aclanthology.org/2022.lrec-1.717/)
- ⭐🌍 07/2022: [A Few Thousand Translations Go a Long Way! Leveraging Pre-trained Models for African News Translation](https://aclanthology.org/2022.naacl-main.223/)
- ⭐🌐 07/2022: [No Language Left Behind: Scaling Human-Centered Machine Translation](https://arxiv.org/abs/2207.04672)
- ⭐🌐 11/2022: [NTREX-128 – News Test References for MT Evaluation of 128 Languages](https://aclanthology.org/2022.sumeval-1.4/)
- ⏫🌐 12/2022: [SMaLL-100: Introducing Shallow Multilingual Machine Translation Model for Low-Resource Languages](https://aclanthology.org/2022.emnlp-main.571/)
- ⏫ 02/2023: [Low-Resourced machine translation for Senegalese Wolof language](https://arxiv.org/pdf/2305.00606)
- 📰 03/2023: [Kàllaama NMT: un ensemble d'outils IA pour rendre le numérique plus inclusif en Afrique](https://youtu.be/P5PRgugOu8o?t=117)
- ⭐🌐 09/2023: [MADLAD-400: A Multilingual And Document-Level Large Audited Dataset](https://arxiv.org/abs/2309.04662)
- 📰🌐 06/2024: [110 new languages are coming to Google Translate](https://blog.google/products/translate/google-translate-new-languages-2024/)
- 📰 12/2024: [LAfricaMobile NMT](https://www.thd.tn/lafricamobile-lance-une-intelligence-artificielle-capable-de-traduire-du-francais-vers-le-wolof-le-bambara-et-dioula/)
- ⭐⏫ 02/2025: [SMOL: Professionally translated parallel data for 115 under-represented languages](https://arxiv.org/abs/2502.12301)
- 📰🌐 11/2025: [Wolof among supported languages in DeepL](https://support.deepl.com/hc/en-us/articles/360019925219-DeepL-Translator-languages)
- 📰 12/2025: [GalsenAI French-Wolof Translator](https://github.com/Galsenaicommunity/Wolof-NMT)
- 📰 11/2025: [CLAD FirilMa Traducteur](https://cladfirilma.ucad.sn/)
- 🌐⏫ 01/2026: [TranslateGemma Technical Report](https://arxiv.org/abs/2601.09012)
- 🌐⏫ 01/2026: [AfriNLLB: Efficient Translation Models for African Languages](https://openreview.net/forum?id=hVJZNUZBur) 🔃
- ⏫🌐 03/2026: [Omnilingual MT: Machine Translation for 1,600 Languages](https://ai.meta.com/research/publications/omnilingual-mt-machine-translation-for-1600-languages/) 🔃

### Question Answering and Dialogue Systems [+LLMs]
- ⏫🌍 05/2022: [AfriWOZ: Corpus for Exploiting Cross-Lingual Transfer for Dialogue Generation in Low-Resource, African Languages](https://arxiv.org/abs/2204.08083)
- ⏫🇫🇷 06/2022: [Preuve de concept d’un bot vocal dialoguant en wolof (Proof-of-Concept of a Voicebot Speaking Wolof)](https://aclanthology.org/2022.jeptalnrecital-taln.40/)
- ⭐🌐 11/2022: [BLOOM: A 176B-Parameter Open-Access Multilingual Language Model](https://arxiv.org/abs/2211.05100) 🔃
- ⭐🌍 12/2022: [AfroLM: A Self-Active Learning-based Multilingual Pretrained Language Model for 23 African Languages](https://aclanthology.org/2022.sustainlp-1.11/)
- 📰 03/2023: [Local Partnership Launches Digital Health Tool to Decrease Hypertension in Senegal](https://www.intrahealth.org/news/local-partnership-launches-digital-health-tool-decrease-hypertension-senegal)
    > More info on: https://saytutension.sante.sn.
- ⏫🌍 05/2023: [AfriQA: Cross-lingual Open-Retrieval Question Answering for African Languages](https://arxiv.org/abs/2305.06897)
- ⏫ 06/2023: [Design, development and usability of an educational AI chatbot for People with Haemophilia in Senegal](https://onlinelibrary.wiley.com/doi/10.1111/hae.14815) 🔃
- ⏫🌍 07/2023: [SERENGETI: Massively Multilingual Language Models for Africa](https://aclanthology.org/2023.findings-acl.97/)
- ⏫🌍 01/2024: [Cheetah: Natural Language Generation for 517 African Languages](https://aclanthology.org/2024.acl-long.691/) 🔃
- ⏫🌍 06/2024: [IrokoBench: A New Benchmark for African Languages in the Age of Large Language Models](https://arxiv.org/abs/2406.03368)
- ⏫🌐 06/2024: [Fumbling in Babel: An Investigation into ChatGPT’s Language Identification Ability](https://aclanthology.org/2024.findings-naacl.274/) 🔃
- ⭐🌐 08/2024: [Aya Dataset: An Open-Access Collection for Multilingual Instruction Tuning](https://aclanthology.org/2024.acl-long.620/)
    > Wolof was the additional language in the Aya dataset that had to be excluded from training [(Üstün et al., 2024)](https://arxiv.org/abs/2402.07827).
- ⏫🌐 08/2024: [The Belebele Benchmark: a Parallel Reading Comprehension Dataset in 122 Language Variants](https://aclanthology.org/2024.acl-long.44/)
- ⏫🌐 08/2024: [Goldfish: Monolingual Language Models for 350 Languages](https://arxiv.org/abs/2408.10441) 🔃
- ⏫ 01/2025: [Task-Oriented Dialog Systems for the Senegalese Wolof Language](https://aclanthology.org/2025.coling-main.322/)
- ⏫🌐🔐 01/2025: [A Comprehensive Von Willebrand Disease Awareness and Support Chatbot for Senegalese Communities](https://ieeexplore.ieee.org/abstract/document/10992905)
- 📰 12/2024: [AWA: Senegalese start-up's AI muse speaks in Wolof](https://www.trtafrika.com/english/article/18244712)
    > A subsequent [Awa-Milkyway](https://www.linkedin.com/search/results/content/?fromMember=%5B%22ACoAADONOG0B7Ul4mepy1frBb_X8AAlK0EJL-44%22%5D&keywords=milkyway&origin=FACETED_SEARCH&sid=28d) model has also been announced but not published since then.
- 📰 01/2025: [Oolel: A High-Performing Open LLM for Wolof](https://huggingface.co/soynade-research/Oolel-v0.1)
- ⏫🌐 04/2025: [MultiBLiMP 1.0: A Massively Multilingual Benchmark of Linguistic Minimal Pairs](https://arxiv.org/abs/2504.02768) 🔃
- ⏫🌍 06/2025: [The State of Large Language Models for African Languages: Progress and Challenges](https://arxiv.org/abs/2506.02280) 
    > Reports that `AfriTeva` and `AfroXLMR` support Wolof but it's not the case, might be a mistake.
- ⏫🌐 07/2025: [Where Are We? Evaluating LLM Performance on African Languages](https://aclanthology.org/2025.acl-long.1572/)
- ⏫🌐 09/2025: [MERLIN: Multi-Stage Curriculum Alignment for Multilingual Encoder-LLM Integration in Cross-Lingual Reasoning](https://arxiv.org/abs/2509.08105) 🔃
- 📰🌐 02/2026: [Tiny AYA: Making Multilingual AI Accessible](https://cohere.com/blog/cohere-labs-tiny-aya) 🔃

### Pre-training corpus
- ⭐🌐 01/2022: [Quality at a Glance: An Audit of Web-Crawled Multilingual Datasets](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00447/109285/Quality-at-a-Glance-An-Audit-of-Web-Crawled)
- ⭐🌐 09/2023: [MADLAD-400: A Multilingual And Document-Level Large Audited Dataset](https://arxiv.org/abs/2309.04662)
- ⏫🌐 06/2025: [FineWeb2: One Pipeline to Scale Them All -- Adapting Pre-Training Data Processing to Every Language](https://arxiv.org/abs/2506.20920)


### Speech Processing

#### Automatic Speech Recognition (ASR)
- ⏫ 04/2011: [Speech Recognition and Text-to-speech Solution for Vernacular Languages](https://personales.upv.es/thinkmind/dl/conferences/icdt/icdt_2011/icdt_2011_3_30_20061.pdf)
- ⏫🌐 09/2015: [Speech Technologies for African Languages: Example of a Multilingual Calculator for Education](https://hal.science/hal-01170505/)
- ⏫🌐 07/2016: [Automatic speech recognition for African languages with vowel length contrast](https://www.sciencedirect.com/science/article/pii/S1877050916300552)
- ⏫🌐 07/2016: [Speed perturbation and vowel duration modeling for ASR in Hausa and Wolof languages](https://hal.science/hal-01350057/)
- ⏫🌐 06/2017: [Machine Assisted Analysis of Vowel Length Contrasts in Wolof](https://arxiv.org/abs/1706.00465)
- ⭐🌐 08/2016: [Collecting Resources in Sub-Saharan African Languages for Automatic Speech Recognition: a Case Study of Wolof](https://aclanthology.org/L16-1611/)
- ⏫🌍 04/2021: [AI4D -- African Language Program](https://arxiv.org/abs/2104.02516)
- 📰 06/2022: [Wav2vec 2.0 with CTC/Attention trained on DVoice Wolof (No LM)](https://huggingface.co/speechbrain/asr-wav2vec2-dvoice-wolof)
- 📰 03/2023: [Kàllaama ASR: un ensemble d'outils IA pour rendre le numérique plus inclusif en Afrique](https://www.youtube.com/watch?v=P5PRgugOu8o&t=89s)
- ⭐🌐 05/2023: [Scaling Speech Technology to 1,000+ Languages](https://arxiv.org/abs/2305.13516)
- ⏫🌍 06/2023: [Towards hate speech detection in low-resource languages: Comparing ASR to acoustic word embeddings on Wolof and Swahili](https://arxiv.org/abs/2306.00410)
- 📰 08/2023: [Wolof Subtitles Generator](https://github.com/lodjim/wolof-subtitle-generator)
- 📰 11/2023: [OpenAI Whisper and Meta MMS models on fula language](https://github.com/cawoylel/windanam)
- ⏫🌐 04/2024: [Kallaama: A Transcribed Speech Dataset about Agriculture in the Three Most Widely Spoken Languages in Senegal](https://arxiv.org/abs/2404.01991)
- ⏫🌐 04/2024: [Africa-centric self-supervised pre-training for multilingual speech representation in a sub-saharan context](https://arxiv.org/abs/2404.02000)
- ⏫🌐 04/2024: [Self-supervised and multilingual learning applied to the Wolof, Swahili and Fongbe](https://inria.hal.science/hal-04547298v3)
- 📰 05/2024: [Senegalese startup Lengo brings AI to informal retailers](https://www.theafricareport.com/347968/senegalese-startup-lengo-brings-ai-to-informal-retailers/)
- ⏫ 06/2024: [State-of-the-Art Review on Recent Trends in Automatic Speech Recognition](https://link.springer.com/chapter/10.1007/978-3-031-63999-9_11)
- ⏫🌐🇫🇷 07/2024: [Représentation de la parole multilingue par apprentissage auto-supervisé dans un contexte subsaharien](https://aclanthology.org/2024.jeptalnrecital-jep.17/)
- 📰 08/2024: [ASR-Africa's Collections - Fula](https://huggingface.co/collections/asr-africa/fula)
- 📰 09/2024: [ASR-Africa's Collections - Wolof](https://huggingface.co/collections/asr-africa/wolof)
- ⏫🌍 11/2024: [Multilingual speech recognition initiative for African languages](https://link.springer.com/article/10.1007/s41060-024-00677-9)
- 📰 11/2024: [Orange to expand open-source AI models to African regional languages for digital inclusion](https://newsroom.orange.com/orange-to-expand-open-source-ai-models-to-african-regional-languages-for-digital-inclusion/)
- 📰 12/2024: [LAfricaMobile STT](https://www.thd.tn/lafricamobile-lance-une-intelligence-artificielle-capable-de-traduire-du-francais-vers-le-wolof-le-bambara-et-dioula/)
- 📰 01/2025: [Caytu Whosper-large-v2](https://huggingface.co/CAYTU/whosper-large-v2)
- ⏫🌍 07/2025: [Synthetic Voice Data for Automatic Speech Recognition in African Languages](https://arxiv.org/abs/2507.17578v1)
- 📰 09/2025: [Breaking Language Barriers in African Healthcare: Fine-Tuning Speech Recognition for Wolof and Hausa in Maternal and Reproductive Health](https://www.linkedin.com/pulse/breaking-language-barriers-african-healthcare-fine-tuning-speech-zzqze/)
    > The poster [can be viewed here](https://drive.google.com/file/d/1Qv8Y7SV0oSJoWjktggdDOoHaXBGAFSAt/view).
- 📰 09/2025: [Benchmarking Automatic Speech Recognition Models for African Languages](https://docs.google.com/presentation/d/1IXWl1rE2UeXTDTbU5ilQOXuL_iUaU3p7VXY0OMvECHI/edit?slide=id.p#slide=id.p)
- ⏫ 09/2025: [WolBanking77: Wolof Banking Speech Intent Classification Dataset](https://arxiv.org/abs/2509.19271)
- ⏫ 09/2025: [Speech Language Models for Under-Represented Languages: Insights from Wolof](https://arxiv.org/abs/2509.15362)
- ⏫🌐 11/2025: [Omnilingual ASR: Open-Source Multilingual Speech Recognition for 1600+ Languages](https://ai.meta.com/research/publications/omnilingual-asr-open-source-multilingual-speech-recognition-for-1600-languages/)
- ⏫🌐 11/2025: [Voice of a Continent: Mapping Africa’s Speech Technology Frontier](https://aclanthology.org/2025.emnlp-main.559/) 🔃
- 📰🌐 12/2025: [ElevenLabs' Scribe v2: Wolof Speech to Text Transcription](https://elevenlabs.io/speech-to-text/wolof) 🔃
- 📰🌍 02/2026: [PazaBench: ASR leaderboard for low-resource languages](https://huggingface.co/spaces/microsoft/paza-bench) 🔃

#### Speech Synthesis / Text To Speech (TTS)
- ⏫ 04/2011: [Speech Recognition and Text-to-speech Solution for Vernacular Languages](https://personales.upv.es/thinkmind/dl/conferences/icdt/icdt_2011/icdt_2011_3_30_20061.pdf)
- 📰 10/2020: [Building Wolof Text To Speech System](https://k4all.org/project/building-wolof-text-to-speech-system/)
- ⏫🌍 07/2022: [Building African Voices](https://arxiv.org/abs/2207.00688)
- 📰 03/2023: [Kàllaama TTS: un ensemble d'outils IA pour rendre le numérique plus inclusif en Afrique](https://youtu.be/P5PRgugOu8o?t=65)
- 📰 09/2024: [Wolof TTS](https://huggingface.co/galsenai/xTTS-v2-wolof)
- 📰 12/2024: [LAfricaMobile TTS](https://lafricamobile.com/en/produit-tts/)
- 📰 02/2025: [Adia_TTS Wolof](https://huggingface.co/CONCREE/Adia_TTS)
- 📰 06/2025: [TTS-WOLOF : Building Inclusive Voice AI for African Languages – The Wolof Case](https://ascii.org.sn/index.php/cnria-2025)
- 📰 03/2026: [Oolel-Voice: a high-quality text-to-speech model purpose-built for Wolof](https://huggingface.co/soynade-research/Oolel-Voices) 🔃

#### Spoken Dialogue Systems
##### Spoken Language Understanding (SLU)
- 📰 09/2020: [Keyword Spotting with African Languages](https://k4all.org/project/keyword-spotting-with-african-languages/)
    > The 1st and only research project that targeted all the 06 main Senegalese languages so far.
- ⏫🇫🇷 06/2022: [Preuve de concept d’un bot vocal dialoguant en wolof (Proof-of-Concept of a Voicebot Speaking Wolof)](https://aclanthology.org/2022.jeptalnrecital-taln.40/)
- ⏫🌍 06/2023: [Towards hate speech detection in low-resource languages: Comparing ASR to acoustic word embeddings on Wolof and Swahili](https://arxiv.org/abs/2306.00410)
- ⏫ 09/2025: [WolBanking77: Wolof Banking Speech Intent Classification Dataset](https://arxiv.org/abs/2509.19271)

##### Speech Language Models (SLMs)
- ⏫ 09/2025: [Speech Language Models for Under-Represented Languages: Insights from Wolof](https://arxiv.org/abs/2509.15362)


### Multi-task Benchmark
- ⏫🌐 05/2023: [XTREME-UP: A User-Centric Scarce-Data Benchmark for Under-Represented Languages](https://arxiv.org/abs/2305.11938)
    > Reports that it covers `ASR`, `NER` and `MT` tasks for Wolof but no Wolof training data has been found in [the dataset](https://github.com/google-research/xtreme-up/) for translation.


## Citation

If this work was useful regarding your research, please cite the paper as:

```bibtex
@article{sen-nlp-survey,
	doi = {10.20944/preprints202601.1124.v1},
	url = {https://doi.org/10.20944/preprints202601.1124.v1},
	year = 2026,
	month = {January},
	publisher = {Preprints},
	author = {Derguene Mbaye and Tatiana D. P. Mbengue and Madoune R. Seye and Moussa Diallo and Mamadou L. Ndiaye and Dimitri S. Adjanohoun and Djiby Sow and Cheikh S. Wade and Jean-Claude B. Munyaka and Jerome Chenal},
	title = {Opportunities and Challenges of Natural Language Processing for Low-Resource Senegalese Languages in Social Science Research},
	journal = {Preprints}
}
```
Feel free to also leave a star 🌟️ 