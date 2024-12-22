[Whisper](https://arxiv.org/pdf/2212.04356), released in 2022, quickly grew to be one of the most popular ASR models and demonstrated the efficacy of the [Transformer](https://arxiv.org/pdf/1706.03762) architecture across media, introducing it to a new domain: speech. Whisper's release was intended to monitor the empirical impact of large, weakly-supervised data on an off-the-shelf model architecture. Previous speech recognition models had either been trained on large, unsupervised data, lacking quality, or small, gold-standard training data, lacking quantity. The novel dataset size was referenced in the authors' begrudging apellation of their work: WSPSR (Whisper) standing for Web-scale Supervised Pretraining for Speech Recognition. 

# Architecture

As mentioned above, the intention behind Whisper was an empirical examination of the impact weakly supervised, web-scale data would have on speech recogition tasks. For this reason, the authors chose a simplified Transformer network and observed a direct correlation between model performance and model size. Below, we'll briefly summarize the model architecture and its similarity to the original Transformer architecture. For those already familiar with the Transformer model, Whisper's encoder and decoder blocks are [pre-activation residual blocks](https://arxiv.org/pdf/1603.05027).

[Attention Is All You Need](https://arxiv.org/pdf/1706.03762), released in 2017, rapidly accelerated machine learning and artificial intelligence development. Five years later, its sequence-to-sequence efficacy led to OpenAI researchers utilizing it for speech. If you haven't read the original Attention paper, I'd urge you to read it over. It's a well-defined and detailed piece of literature explaining the authors' motivations, experiments, and findings. It directly correlates to the Whisper architecture, with only a slight difference to allow audio compatibility.

## Encoder

The first step for any encoder is transforming the input media into vectors. In the original Transformer architecture (text domain), this was implemented through tokenization and text embeddings. For audio, it entailed a log-mel-spectrogram and convolutional audio stem. Log-mel spectrograms serve to better capture audio information than simple frequency information. The encoder's responsibility is encoding the input information into the most important audio features. The convolutional stem processes the log-mel spectrogram and transforms it to be dimensionally compatible with the remainder of the model architecture. Log-mel spectrograms are not critical to understanding  Whisper, but I've included a small section at the [bottom of this page](#log-mel-spectrogram) with more information on them, or you can check out [this video](https://www.youtube.com/watch?v=9GHCiiDLHQ4) which does a great job of explaining the concept. 

The encoder stem is made of two successive convolutional layers each followed by a GELU activation function. The first convolutional layer has a 3x3 kernel with stride of 1 and padding of 1. After passing through the subsequent activation function, the second convolutional layer also has a 3x3 kernel with a stride of 2 and padding of 1. Audio features pass through the second GELU function before progressing to the bulk of the audio encoder. A simple diagram is illustrated below.

<p align="center" width="100%">
  <img src="/Images/encoder_stem_diagram.png" width="100%">
</p>

Like any attention-dependent neural network, the relevant embeddings have to understand their input position. Whisper uses a fixed sinusoidal positional encoding for the audio encoder. The stem output is summed with the pertinent position information before progressing to the bulk of the encoder block, seen below. Identical to the original Transformer architecture, these encoder blocks contain two sub-layers. A residual connection exists around each sub-layer, represented in the diagram below as the green arrows below the operational blocks. The propagated signal is summed with the residual immediately after each sub-layer. In the first sub-layer, audio features are immediately normalized before multi-head self-attention is applied. The result is summed with the first sub-layer residual. The second sub-layer is a feed-forward network, with two linear projections sandwiching a GELU activation function. After progressing through the feed-forward network, the result is then summed with the second sub-layer residual. The output of each encoder block goes to two locations. First, it feeds into the next block, serving as the adjacent encoder block's input. Second, it serves as the key and value vectors for the cross-attention mechanism in the corresponding decoding block. More details on self- and cross-attention will be provided later.

<p align="center" width="100%">
  <img src="/Images/encoder_diagram.png" width="100%">
</p>

## Decoder

One of the issues with previous audio speech recognition architectures utilizing large, unlabeled datasets was the lack of a performant decoder. The encoders learned high-quaity representations of speech, but the models required finetuning on specific datasets for competent decoders. This leaves them vulnerable to dataset specific quirks when fine-tuning, limiting the model's generalizability to other data. To combat the lack of similarly performant decoders, Whisper researchers employed Transformer blocks for both the encoder and decoder. Training on web-scale data, this created a more robust architecture capable of generalizing zero-shot to out-of-distribution audio.

The architecture for a singular decoding block can be seen in the illustration below. It follows identically to the encoding block architecture, with two distinctions: unlike the encoder, which employed fixed sinusoidal position embeddings, the decoder utilizes learned positional embeddings, and the architecture is augmented to allow for cross-attention. This gives the decoding block three sub-layers, still with residual connections around each. First, features are normalized before self-attention is applied and summed with the residual connection. Next, they are normalized before cross-attention is applied and summed with its residual. Third, they're normalized before propagating through the feed-forward network and again, summing with the residual connection. The output of each decoding block progresses to become the input for the next decoding block.

<p align="center" width="100%">
  <img src="/Images/decoder_diagram.png" width="100%">
</p>

Stacking the blocks described above determines the model complexity. Larger, more complex models lead to better performance. However, larger models are also more susceptible to hallucination. Interestingly,  researchers were able to create a competent Whisper model with only 4 encoder and decoder blocks each, referred to as the Tiny model. The base Whisper model employed 6 encoding and decoding blocks, the same number as the original Transformer architecture. Researchers' most successful model required 32 encoding and decoding blocks, but offered optimal zero-shot transcription, translation, and language recognition performance. It's important to note that the Whisper training paradigm was designed to be multilingual and multitask. At smaller model sizes, this led to negative transfer. The same model parameters trained with English-only data outperformed their multilingual counterparts. At larger model sizes, this trend disappeared, with multilingual models outperforming their English-only complements. The different model sizes available at the time of the paper's release can be seen below.

<p align="center" width="100%">
  <img src="/Images/model_sizes.png" width="50%">
</p>

The decoder's responsibility, as an audio-conditional language model, is to transform audio features to text. It does this iteratively, predicting the next token given the previous tokens. The Whisper decoding sequence always begins the same way, illustrated below. This special token prefix to text generation consolidated the one-to-many mapping responsibilities of an automated speech recognition model, offering a simple user interface to specify the model task. Intuitively, it also mitigates the hallucination problem. Language models have well-documented issues with hallucination, but utilizing a consistent prefix emphasizes a coherent output and limits the risk of nonsensical text generation.

<p align="center" width="100%">
  <img src="/Images/decoding_pipeline.png" width="100%">
</p>

The choice of decoding strategy can also inhibit hallucinations and repetitive text generation. Whisper implements two decoding strategies: greedy and beam search decoding. Greedy decoding endlessly generates the next most probable token. As a result, it is susceptible to repetitive looping. This can be controlled through temperature. Temperature is a parameter determining which token is selected next, ranging from 0 to 1. A temperature of 0 always selects the next most likely token, while 1 introduces variability into the token generation. Researchers implemented two thresholds to determine temperature: log probability and compression rate. Log probability is the cumulative sum of each token in the path's probability. Compression rate computes the ratio of repeated text in the generated sequence. Starting at a temperature of 0, if the generated tokens fall below either of the thresholds, it is increased by 0.2. Beam search decoding maintains the n most probable token generation paths, determined by log-probability. Ultimately, the decoding path with the highest log-probability is selected as the output. The authors employed beam search decoding with n=5.

## Attention

Attention serves as a quantification of the compatibility between vector representations of information. Given a query vector and a set of key-value vector pairs, attention assigns a weight to each value. Each value's respective weight is a measure of the congruity between the corresponding query and key. There are a few different attention implementations, but the most popular (and the one implemented for Transformer and Whisper) is scaled dot-product attention. The scaled dot-product attention equation can be seen below. Attention can either be implemented to solidify the model's understanding of an information vector and its context or to exchange information across two data types and unify the model's understanding of both. These respective methods are known as self-attention and cross-attention.

<p align="center" width="100%">
  <img src="/Images/sdpa_attn_equation.png" width="50%">
</p>

### Self-Attention

Self-attention is implemented in both the encoder and decoder. It's designed to compound the network's understanding of the propagated signal. In the encoder, it takes as input the propagated query, key, and value vectors from the previous encoding block. The encoder attends to all of the positions from the previous layer. Allowing the network to understand all of the information, across all positions, helps it detect and emphasize the most important features. In Whisper, this means detecting and encoding the most important audio features.

In the decoder, self-attention is nearly identical. It attends to all positions up to the current position. The auto-regression of Transformers hinges on predicting the next token based on the currently available information. Information ahead of the current position should not affect the current token output, so information beyond the current position is masked out. The reason for this is that, at inference, all of the audio information is available so all of it should be attended to in the encoder. The model doesn't have the corresponding text transcription. Its responsibility is to sequentially predict each word corresponding to the provided audio. Since textual information is only provided up to the current token at inference time, it's trained with the same philosophy.

### Cross-Attention

Cross-attention exists as the information crossover between the encoder and the decoder. It functions identically to self-attention, quantifying the similarity between a query and key vector. Unlike in self-attention, the query, key, and value vectors come from different locations. The decoding block provides the query vector, while the key and value vectors are supplied by the corresponding encoding block. This allows every position in the decoding vector to attend to every position in the encoding vector, providing key context, and bridging the gap between encoded audio features and autoregressively decoded text.

# Engineering

The intention behind Whisper was monitoring the impact of large-scale weakly supervised data on an off-the-shelf sequence-to-sequence model architecture. Despite this simple idea, there's always more going on behind the curtain to create a successful model. Below, we'll take a look at some of the engineering details that OpenAI researchers implemented to create such a performant model. They'll be split into two sections: the first will focus on heuristics employed in the training stage to ensure successful pre-training and the second will focus on optimizations that popularized Whisper for its inference speed.

## Attention to Detail in Training

### Data Filtering

Despite training on nearly 700k hours of audio, the authors ensured the quality of their data through multiple data filtering heuristics. Training the sequence-to-sequence task of audio transcription required a dataset of pairs containing the audio and the transcript. Researchers quickly detected that many of the transcripts found online were machine-generated outputs. Researchers ruled these out to prevent Whisper from learning "transcript-ese", mimicking the output of transcription models as opposed to natural language. By removing transcripts lacking punctuation, in all uppercase or in all lowercase, researchers ensured Whisper was trained only on human annotated transcripts. 

Whisper was always intended to be multitask and multilingual. Providing quality multilingual data required additional tuning. Researchers used an earlier iteration of the Whisper model trained on a precursor version of the dataset solely to serve as a language detector. Given the pair of an audio and its transcript, researchers used the prototype model to detect the audio language and [CLD2](https://github.com/CLD2Owners/cld2) to detect the transcript language. If the two languages matched, the audio-transcript pair was included in the dataset as a sample of language X. If the two languages differed, but the transcript language was English, researchers included the pair as a training example for the X -> EN task, responsible for translating unknown audio to English. Otherwise, the audio and its transcript was discarded and not included in the dataset. Given the potential for overlapping transcriptions, maybe the Spanish transcript of the Lion King was exceedingly popular, the authors utilized fuzzy deduplication to reduce redundant duplicate transcripts. 

After training their prototype model, researchers analyzed data sources with the highest error rate and manually inspected these sources. This uncovered multiple partially or poorly transcribed texts the model was struggling to align with the corresponding audio. Additionally, it revealed low-quality machine generated transcripts that had escaped the authors' detection heuristics. After finding these low-quality transcripts, the authors removed them, improving their training data for the next model version.

Machine learning is cyclic and iterative. Despite the steps taken by researchers to remove low-quality data from their training set, there's always more to be done. Researchers measured the relationship between the model’s performance in a certain language and the quantity of training data available in that language. They found that the model significantly underperformed in Welsh and, upon closer inspection, found that the majority of the 9k hours of Welsh audio the model was trained on was incorrectly labeled English audio. It emphasizes, once again, the bidirectional relationship between training data and models. For every newer, better, model iteration, there is always some way the data can be improved, leading to further model improvements, continuing the cycle. 

### Text Standardization

Researchers opted against including a text normalization step in their model training pipeline. They relied on the Transformer architecture successfully capturing the sequence-to-sequence relationship between audio and text. This allowed Whisper to capture the natural language text of transcripts and, given the size of the data, prevented Whisper from overfitting to any specific transcript formatting. The result was an audio speech recognition model capable of producing natural language from audio without being reliant on excessive standardization to produce coherent results.

Of course, evaluation is a different game. Researchers manually inspected where their predicted transcriptions were penalized by word error rate (WER) and developed a text normalizer for inference to combat these issues. [Examples](https://github.com/openai/whisper/blob/main/whisper/normalizers/english.py) include identifying multi-digit numbers, converting spelled-out numbers to digits, and correcting currency symbols to natural language. Given the fear that their normalizer was overcorrecting vulnerabilities in Whisper's inference, researchers evaluated the same model's performance on the same evaluation sets with a different, [open-source text normalizer](https://www.pnas.org/doi/full/10.1073/pnas.1915768117#sec-3). Both results achieved roughly even scores. They found performance distinctions arose from trivialities in WER scoring, including allowing contractions and spelling out numerical or monetary expressions.

### Text Conditioning for Audio-Conditional Decoder

As mentioned in the architecture section, the decoder is an audio-conditional language model. It conditions on audio features from the encoder via cross-attention implemented in the Transformer blocks. Researchers also conditioned the decoder on text to provide greater context than the 30s of audio available to the encoder. The previous text history was provided in the hopes it would help navigate unclear audio, especially if important context was outside of the current 30s window. Given a previous conversation regarding favorite books, it becomes much easier to decide if the spoken word was "read" or "red". Researchers also found that this text conditioning improved transcription performance with greedy decoding when temperature was below 0.5.

Prompt engineering was quickly popularized as a method to deliver the best results from large language models (LLMs) and limit hallucinations. As the size of Whisper models grew, so did the potential for hallucination. Unreferenced in the paper, but available in the code repository, is an [option to prompt Whisper models](https://github.com/openai/whisper/blob/main/whisper/transcribe.py#L101), providing them a narrower scope to consider when transcribing audio. Functioning similar to prompting for LLMs, this offers the opportunity to provide a specific focus to the model in decoding the audio.

Since many of the audio and transcript pairs the model was trained on included multiple speakers, these transcripts would often include speaker diarization. Given Whisper's training responsibility of accurately matching its transcription to the provided text, the model would also often attempt to predict speaker names. That's a nearly impossible challenge given sliding windows containing 30s of audio each. Naturally, this led to many incorrect guesses on the current speaker. Researchers recognized this shortcoming and finetuned the model on a subset of transcripts entirely lacking speaker annotations to train speaker diarization out of the model.

## Inference Optimizations

### Caching

Caching is a core tenet of software engineering, utilized to accelerate repetitive function calls by storing their results for easy access. Whisper caches attention mechanism computations that would otherwise be repeatedly recalculated. It implements this through [hooks](https://www.youtube.com/watch?v=syLFCVYua6Q) on attention layers within the model, allowing intermediate values to be recorded for future calls. The primary beneficiary here is cross-attention. Encoded audio features interact with their decoded tokenized counterparts through the cross-attention mechanism in the decoder blocks, serving as the key and value vectors to the decoder-supplied query vector. Encoded features are immutable. This means that, for each encoding block, the same computation is regularly recalculated to supply the key and value vectors for cross-attention. It would be recalculated for every sequential, auto-regressively predicted token. Recognizing this inefficiency, researchers cached the attention projection results, offering faster access and significantly accelerating inference.

Whisper also uses Python's [functools library](https://docs.python.org/3/library/functools.html) to cache repeated calculations. The decorator
```
@lru_cache(maxsize) 
```
offers an easy instruction to cache the most recent function return values until the specified maxsize. It operates on basic LRU cache principles, expelling the least recently used return value to make space for unique function calls. A maxsize of 1, storing only the most recent return value, is similar to another decorator:
```
@cached_property
```
which caches a singular return value for a function without variable input. These two speed-ups are most effectively employed for Whisper's [tokenizer](https://github.com/openai/whisper/blob/main/whisper/tokenizer.py), responsible for converting numeric values to text. Caching the most frequently called tokens prevents Whisper from reconverting text to its numerical representation, effortlessly speeding up the decoding process.

### Just In Time (JIT) Compilation

# Log-Mel Spectrogram

# Conclusion


This section isn't really relevant to the paper, but I think it's parallel. The greatest success of OpenAI since its inception was its emphasis on zero-shot evaluation. If you study machine learning in academia, at least the way I learned it, you're given a training set and a testing set. Maybe you're asked to perform cross-validation and maybe you use a validation set but, odds are likely you're taught to split some quantity of data into 80% for training and 20% for testing (as an example, percentage splits can obviously vary). The problem with this, as [observed in section 3.3 of the Whisper paper](https://cdn.openai.com/papers/whisper.pdf#page=5) is that you're really evaluating the model's ability to capture in-distribution trends. All of the training and testing data, provided they come from the same source, come from the same distribution. Your model shouldn't be evaluated on its ability to capture in-distribution trends, especially when neural networks have demonstrated a [propensity for picking up data trends invisible to the human eye](https://arxiv.org/pdf/2103.00020). Robust models should generalize well to out-of-distribution data and that should be the basis of their evaluation. This is a significant motivation behind the evaluation of Whisper and underpins OpenAI's other model releases including Dall-E and GPT. This all obviously comes with the acknowledgement that it is researchers' responsibility to prevent data contamination and ensure that zero-shot evaluation is in fact evaluation on previously unseen data.
