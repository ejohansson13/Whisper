[Whisper](https://arxiv.org/pdf/2212.04356), released in 2022, quickly grew to be one of the most popular ASR models and demonstrated the efficacy of the [Transformer](https://arxiv.org/pdf/1706.03762) architecture across media, introducing it to a new domain: speech. Whisper's release was intended to monitor the empirical impact of large, weakly-supervised data on an off-the-shelf model architecture. Previous speech recognition models had either been trained on large, unsupervised data, lacking quality, or small, gold-standard training data, lacking quantity. The novel dataset size was referenced in the authors' begrudging apellation of their work: WSPSR (Whisper) standing for Web-scale Supervised Pretraining for Speech Recognition. 

# Architecture

As mentioned above, the intention behind Whisper was an empirical examination of the impact weakly supervised, web-scale data would have on speech recogition tasks. For this reason, the authors chose a simplified Transformer network and observed a direct correlation between model performance and model size. Below, we'll briefly summarize the model architecture and its similarity to the original Transformer architecture. For those already familiar with the Transformer model, Whisper's encoder and decoder blocks are [pre-activation residual blocks](https://arxiv.org/pdf/1603.05027).

[Attention Is All You Need](https://arxiv.org/pdf/1706.03762), released in 2017, rapidly accelerated machine learning and artificial intelligence development. Five years later, its sequence-to-sequence efficacy led to OpenAI researchers utilizing it for speech. If you haven't read the original Attention paper, I'd urge you to read it over. It's a well-defined and detailed piece of literature explaining the authors' motivations, experiments, and findings. It directly correlates to the Whisper architecture, with only a slight difference to allow audio compatibility.

## Encoder

The first step for any encoder is transforming the input media into vectors. In the original Transformer architecture (text domain), this was implemented through tokenization and text embeddings. For audio, it entailed a log-mel-spectrogram and convolutional audio stem. Log-mel spectrograms serve to better capture audio information than simple frequency information. The encoder's responsibility is encoding the input information into the most important audio features. The convolutional stem processes the log-mel spectrogram and transforms it to be dimensionally compatible with the remainder of the model architecture. Log-mel spectrograms are not critical to understanding  Whisper, but I've included a small section at the [bottom of this page](link here) with more information on them, or you can check out [this video](https://www.youtube.com/watch?v=9GHCiiDLHQ4) which does a great job of explaining the concept. 

The encoder stem is made of two successive convolutional layers each followed by a GELU activation function. The first convolutional layer has a 3x3 kernel with stride of 1 and padding of 1. After passing through the subsequent activation function, the second convolutional layer also has a 3x3 kernel with a stride of 2 and padding of 1. Audio features pass through the second GELU function before progressing to the bulk of the audio encoder. A simple diagram is illustrated below.

<p align="center" width="100%">
  <img src="/Images/encoder_stem_diagram.png" width="100%">
</p>

Like any attention-dependent neural network, the relevant embeddings have to understand their input position. Whisper uses a fixed sinusoidal positional encoding for the audio encoder. The stem output is summed with the pertinent position information before progressing to the bulk of the encoder block, seen below. Identical to the original Transformer architecture, these encoder blocks contain two sub-layers. A residual connection exists around each sub-layer, represented in the diagram below as the green arrows below the operational blocks. The propagated signal is summed with the residual immediately after each sub-layer. In the first sub-layer, audio features are immediately normalized before multi-head self-attention is applied. The result is summed with the first sub-layer residual. The second sub-layer is a feed-forward network, with two linear projections sandwiching a GELU activation function. After progressing through the feed-forward network, the result is then summed with the second sub-layer residual. The output of each encoder block goes to two locations. First, it feeds into the next block, serving as the adjacent encoder block's input. Second, it serves as the key and value vectors for the cross-attention mechanism in the corresponding decoding block. More details on self- and cross-attention will be provided later.

<p align="center" width="100%">
  <img src="/Images/encoder_diagram.png" width="100%">
</p>

## Decoder

One of the issues with previous audio speech recognition architectures utilizing large, unlabeled datasets was the lack of a performant decoder. The encoders learned high-quaity representations of speech, but the models required finetuning on specific datasets for competent decoders. This leaves them vulnerable to dataset specific quirks when fine-tuning, limiting the model's generalizability to other data. To combat the lack of similarly performant decoders, Whisper researchers understood speech recognition as a sequence-to-sequence problem and employed Transformer blocks for both the encoder and decoder. With the web-scale data employed at training time, this created a more robust architecture capable of generalizing zero-shot to out-of-distribution audio.

The architecture for a singular decoding block can be seen in the illustration below. It follows identically to the encoding block architecture, with two distinctions: unlike the encoder, which employed fixed sinusoidal position embeddings, the decoder utilizes learned positional embeddings and the architecture is augmented to allow for cross-attention. This gives the decoding block three sub-layers, still with residual connections around each. First, features are normalized before self-attention is applied and summed with the residual connection. Next, they are normalized before cross-attention is applied and summed with its residual. Third, they're normalized before propagating through the feed-forward network and again, summing with the residual connection. The output of each decoding block progresses to become the input for the next decoding block.

<p align="center" width="100%">
  <img src="/Images/decoder_diagram.png" width="100%">
</p>

Stacking the blocks described above determines the model complexity. Larger, more complex models lead to better performance. However, larger models are also more susceptible to hallucination. Interestingly,  researchers were able to create a competent Whisper model with only 4 encoder and decoder blocks each, referred to as the Tiny model. The base Whisper model employed 6 encoding and decoding blocks, the same number as the original Transformer architecture. Researchers' most successful model required 32 encoding and decoding blocks, but offered optimal zero-shot transcription, translation, and language recognition performance. It's important to note that the Whisper training paradigm was designed to be multilingual and multitask. At smaller model sizes, this led to negative transfer. The same model parameters trained with English-only data outperformed their multilingual counterparts. At larger model sizes, this trend disappeared, with multilingual models outperforming their English-only complements. The different model sizes available at the time of the paper's release can be seen below.

<p align="center" width="100%">
  <img src="/Images/model_sizes.png" width="50%">
</p>

The decoder's responsibility, as an audio-conditional language model, is to transform audio features to text. It does this iteratively, predicting the next token given the previous tokens. The Whisper decoding sequence always begins the same way, illustrated below. The intent for this prefix to textual generation is consolidating the responsibilities of an automated speech recognition model. Specifying the relevant task through tokens greatly simplifies the one-to-many mapping of the Whisper architecture. Intuitively, it also limits hallucinations. Language models have well-documented issues with hallucination, but utilizing a consistent lead-in weights a coherent output and limits the potential for nonsensical or repetitive text generation. 

<p align="center" width="100%">
  <img src="/Images/decoding_pipeline.png" width="100%">
</p>

The choice of decoding strategy can also inhibit hallucinations and repetitive text generation. Whisper implements two strategies: greedy decoding and beam search decoding. Greedy decoding endlessly generates the next most probable token. As a result, it is susceptible to repetitive looping. This can be controlled through temperature. Temperature is a parameter determining which token is selected next, ranging from 0 to 1. A temperature of 0 always selects the next most likely token, while 1 introduces variability into the token generation. Researchers implemented two thresholds to determine temperature: log probability and compression rate. Log probability is the cumulative sum of each token in the path's probability. Compression rate computes the ratio of repeated text in the generated sequence. Starting at a temperature of 0, if the generated tokens fall below either of the thresholds, it is increased by 0.2. Beam search decoding maintains the n most probable token generation paths, determined by log-probability. Ultimately, the decoding path with the highest log-probability is selected as the output. The authors employed beam-search decoding with n=5.

## Attention

Attention serves as a quantification of the compatibility between vector representations of information. Given a query vector and a set of key-value vector pairs, attention assigns a weight to each value. Each value's respective weight is a measure of the congruity between the corresponding query and key. There are a few different attention implementations, but the most popular (and the one implemented for Transformer and Whisper) is scaled dot-product attention. The scaled dot-product attention equation can be seen below.

<p align="center" width="100%">
  <img src="/Images/sdpa_attn_equation.png" width="50%">
</p>

### Self-Attention

Self-attention is implemented in both the encoder and decoder. It's designed to compound the network's understanding of the propagated signal. In the encoder, it takes as input the propagated query, key, and value vectors from the previous encoding block. The encoder attends to all of the positions from the previous layer. Allowing the network to understand all of the information, across all positions, helps it detect and emphasize the most important features. In Whisper, this means detecting and encoding the most important audio features.

In the decoder, self-attention is nearly identical. It attends to all positions up to the current position. The auto-regression of Transformers hinges on predicting the next token based on the currently available information. Information ahead of the current position should not affect the current token output, so information beyond the current position is masked out. The reason for this is that, at inference, all of the audio information is available so all of it should be attended to in the encoder. The model doesn't have the corresponding text transcription. Its responsibility is to sequentially predict each word corresponding to the provided audio. Since textual information is only provided up to the current token at inference time, it's trained with the same philosophy.

### Cross-Attention

Cross-attention exists as the information crossover between the encoder and the decoder. It functions identically to self-attention, quantifying the similarity between a query and key vector. Unlike in self-attention, the query, key, and value vectors come from different locations. The decoding block provides the query vector, while the key and value vectors are supplied by the corresponding encoding block. This allows every position in the decoding vector to attend to every position in the encoding vector, providing key context, and bridging the gap between encoded audio features and autoregressively decoded text.


Could venture into beam search decoding here. 

# Engineering

## Attention to Detail

## Optimizations
