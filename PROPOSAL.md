# Ordinal Multi-Modal Deep Learning for Alzheimer's Disease Severity Prediction

**Ibe Mohammed Ali, Kubra Sag, Poorav Rawat**

---

## Project Type

This is a research-flavor project. We develop a novel ordinal multi-task deep learning framework for Alzheimer's disease (AD) severity prediction using multi-modal biomedical data. The core reference frameworks are CORAL ordinal regression ([Cao et al., 2020](https://doi.org/10.1016/j.patrec.2020.09.024)), homoscedastic multi-task uncertainty weighting ([Kendall et al., 2018](https://arxiv.org/abs/1705.07115)), and discrete-time survival analysis ([Gensheimer & Narasimhan, 2019](https://doi.org/10.7717/peerj.6257)). We extend these methods into a unified cross-cohort system that has not been explored in the AD literature.

## Machine Learning Problem

The problem is formulated as supervised ordinal regression with auxiliary survival prediction. Given a structural MRI brain scan or a speech recording from a cognitive assessment, the model must predict an ordered Alzheimer's disease severity stage (CDR 0, 0.5, 1.0, or 2.0+). For patients at the MCI stage (CDR 0.5), the model additionally predicts a time-varying probability of conversion to AD dementia over a 36-month horizon. Unlike standard multi-class classification, the ordinal formulation explicitly penalizes distant misclassifications more than adjacent ones, respecting the progressive nature of the disease.

## Goals and Motivation

Convolutional neural networks are widely used for MRI-based Alzheimer's diagnosis, but most existing approaches treat disease stages as independent categories using softmax classification ([Qiu et al., 2020](https://doi.org/10.1093/brain/awaa137)). This is problematic for several reasons. First, misclassifying a cognitively normal patient as having moderate dementia is clinically far worse than misclassifying them as very mild, yet cross-entropy loss treats both errors equally. Second, single-task binary models (AD vs. CN) discard the ordinal structure entirely and cannot model disease progression. Third, most multimodal approaches require paired data from the same subjects, discarding the majority of available unimodal data.

Our system addresses these challenges by using CORAL ordinal regression to preserve severity ordering with learned thresholds that map a continuous severity score to ordered stages; jointly training a discrete-time survival head for MCI-to-AD conversion prediction that handles right-censored subjects natively; integrating structural MRI and speech biomarkers from separate cohorts without requiring paired subjects, using cross-cohort alignment via shared ordinal heads and class-conditioned Maximum Mean Discrepancy (MMD) ([Gretton et al., 2012](https://jmlr.org/papers/v13/gretton12a.html)); and balancing multiple task losses automatically through learned homoscedastic uncertainty weights. No prior work combines all four of these elements.

## Methodology and Models

Structural MRI volumes (128x128x128 voxels) are processed by a 3D ResNet-18 ([He et al., 2016](https://arxiv.org/abs/1512.03385)), adapted from the video recognition domain to single-channel medical imaging. The backbone produces a 256-dimensional embedding per scan. For patients with multiple longitudinal visits, per-visit embeddings are aggregated by a GRU recurrent network with sinusoidal time encoding to capture disease trajectory over irregular visit intervals.

Speech recordings from a Cookie Theft picture description task are represented as a concatenation of four feature streams: wav2vec 2.0 embeddings (768-D) ([Baevski et al., 2020](https://arxiv.org/abs/2006.11477)), Sentence-BERT transcript embeddings (384-D), handcrafted acoustic features (216-D including MFCCs, prosody, voice quality, and temporal measures), and handcrafted linguistic features (14-D including lexical diversity, syntactic complexity, semantic coherence, and fluency) ([Fraser et al., 2016](https://doi.org/10.3233/JAD-150520)). This 1,382-D vector is projected to 256-D through a two-layer MLP with LayerNorm and GELU activation.

Both encoders feed into a shared CORAL ordinal head that predicts a single continuous severity score, which three learnable thresholds partition into four ordered CDR stages. The CORAL loss computes binary cross-entropy at each threshold, ensuring rank consistency. For MCI patients, a separate survival head predicts interval-specific hazard probabilities over six 6-month intervals spanning a 36-month window, handling right-censored subjects without imputation. Since MRI and speech data come from non-overlapping populations, the shared CORAL head implicitly aligns representations while class-conditioned MMD explicitly minimizes distributional distance between same-severity embeddings from different modalities. All task losses are combined via homoscedastic uncertainty weighting, where learned variance parameters automatically downweight noisier tasks.

The project follows a structured schedule. Weeks 1-2 focus on data acquisition, MRI preprocessing, and speech feature extraction. Weeks 3-4 involve training unimodal baselines including an MRI ordinal CNN, a speech MLP, and a standard softmax CNN. Weeks 5-6 are dedicated to implementing CORAL ordinal regression and comparing against the softmax baseline. Weeks 7-8 cover building the multi-task model with ordinal and survival heads and implementing cross-cohort MMD alignment. Weeks 9-10 focus on full system training, hyperparameter tuning, and ablation studies. Week 11 is reserved for final evaluation, statistical analysis, and report writing.

## Datasets

The primary imaging dataset is SCAN (NACC), which contains approximately 29,000 3D T1-weighted MRI scans from roughly 10,000 subjects with longitudinal follow-up. Each scan is linked to the NACC Uniform Data Set providing CDR scores, diagnosis codes, and neuropsychological data. SCAN's centralized acquisition and quality control pipeline eliminates the site-harmonization confounds present in ADNI-based studies.

The speech dataset is the DementiaBank Pitt Corpus, which contains approximately 550 Cookie Theft picture description recordings from around 270 subjects (170 probable AD and 100 controls), providing audio files and CHAT-format transcripts with clinical staging. We will additionally evaluate on the Kaggle Alzheimer's Multiclass MRI Dataset (approximately 44,000 augmented 2D MRI slices across 4 classes) to benchmark our ordinal approach against standard multi-class CNN baselines. Patient-level splits are enforced across all datasets to prevent data leakage, with stratification on baseline CDR and conversion status.

## Evaluation

Model performance will be evaluated using Mean Absolute Error (MAE) to measure ordinal stage distance, Quadratic Weighted Kappa (QWK) to assess agreement while penalizing large ordinal errors, and classification accuracy as a standard reference metric. For the survival component, we use Harrell's C-index for ranking accuracy and time-dependent AUC at 12, 24, and 36 months for conversion prediction at clinical time horizons. Baseline comparisons include a standard multi-class CNN with softmax loss, unimodal MRI-only and speech-only models, multi-task training without ordinal constraints, and multi-modal training without cross-cohort alignment. Experiments will vary training set size (10%, 25%, 50%, 75%, 100%) to analyze data efficiency.

## Resources

Training will be performed on Google Colab Pro using a T4 GPU with 16 GB VRAM. Phase 1 (MRI pretraining) requires approximately 9 hours, and Phase 2 (multi-task training) requires approximately 50 hours across 3-4 sessions with checkpoint-resume. The software stack includes Python, PyTorch, torchvision, scikit-learn, nibabel, Parselmouth, and the transformers library for wav2vec 2.0 and Sentence-BERT. Version control is managed through GitHub.

## Workload Distribution

Ibe Mohammed Ali is responsible for the MRI encoder, 3D ResNet-18 adaptation, longitudinal temporal module, cross-cohort alignment via MMD, and system integration. Kubra Sag is responsible for the speech encoder, acoustic and linguistic feature extraction, wav2vec 2.0 and SBERT pipelines, and DementiaBank preprocessing. Poorav Rawat is responsible for the ordinal regression (CORAL) implementation, survival head, multi-task loss balancing, evaluation metrics, and baseline comparisons. All members collaborate on experimental design, ablation studies, and final report writing.

## References

Cao, W., Mirjalili, V., & Raschka, S. (2020). [Rank consistent ordinal regression for neural networks.](https://doi.org/10.1016/j.patrec.2020.09.024) *Pattern Recognition Letters*, 140, 325-331. Kendall, A., Gal, Y., & Cipolla, R. (2018). [Multi-task learning using uncertainty to weigh losses.](https://arxiv.org/abs/1705.07115) *CVPR 2018*. Gensheimer, M. F., & Narasimhan, B. (2019). [A scalable discrete-time survival model for neural networks.](https://doi.org/10.7717/peerj.6257) *PeerJ*, 7, e6257. He, K., Zhang, X., Ren, S., & Sun, J. (2016). [Deep residual learning for image recognition.](https://arxiv.org/abs/1512.03385) *CVPR 2016*. Gretton, A., et al. (2012). [A kernel two-sample test.](https://jmlr.org/papers/v13/gretton12a.html) *JMLR*, 13, 723-773. Qiu, S., et al. (2020). [Interpretable deep learning for Alzheimer's classification.](https://doi.org/10.1093/brain/awaa137) *Brain*, 143(6), 1920-1933. Baevski, A., et al. (2020). [wav2vec 2.0: Self-supervised learning of speech representations.](https://arxiv.org/abs/2006.11477) *NeurIPS 2020*. Fraser, K. C., et al. (2016). [Linguistic features identify Alzheimer's disease in narrative speech.](https://doi.org/10.3233/JAD-150520) *J. Alzheimer's Disease*, 49(2).
