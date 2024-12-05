[Whisper](https://arxiv.org/pdf/2212.04356), released in 2022, quickly grew to be one of the most popular ASR models and demonstrated the efficacy of the [Transformer](https://arxiv.org/pdf/1706.03762) architecture across media, introducing it to a new domain: speech. Whisper's release was intended to monitor the empirical impact of large, weakly-supervised data on an off-the-shelf model architecture. Previous speech recognition models had either been trained on large, unsupervised data, lacking quality, or small, gold-standard training data, lacking quantity. The novel dataset size was referenced in the authors' begrudging apellation of their work: WSPSR (Whisper) standing for Web-scale Supervised Pretraining for Speech Recognition. 

# Architecture

As mentioned above, the intention behind Whisper was an empirical examination of the impact weakly supervised, web-scale data would have on speech recogition tasks. For this reason, the authors chose a simplified Transformer network and observed a direct correlation between model performance and model size. Below, we'll briefly summarize the model architecture and its similarity to the original Transformer architecture. For those already familiar with the Transformer model, the encoder and decoder are made up of [pre-activation residual blocks](https://arxiv.org/pdf/1603.05027).

[Attention Is All You Need](https://arxiv.org/pdf/1706.03762), released in 2017, rapidly accelerated machine learning and artificial intelligence development. Five years later, its sequence-to-sequence efficacy led to OpenAI researchers utilizing it for speech. If you haven't read the original Attention paper, I'd urge you to read it over. It's a well-defined, detailed piece of literature explaining the author's motivations, experiments, and findings. It directly correlates to the Whisper architecture, with only a slight difference to allow audio compatibility.

## Encoder

The first step for any encoder is transforming the input media into vectors. In the original Transformer architecture (text domain), this was implemented through tokenization and text embeddings. For audio, it entailed a log-mel-spectrogram and convolutional audio stem. Log-mel spectrograms serve to better capture audio information than simple frequency information and the convolutional stem processes this information and transforms it to be dimensionally compatible with the remainder of the model architecture. Log-mel spectrograms are not critical to understanding  Whisper, but I've included a small section at the bottom of this page with more information on them, or you can check out [this video](https://www.youtube.com/watch?v=9GHCiiDLHQ4) which does a great job of explaining the concept. 

The encoder stem is made of two successive convolutional layers each followed by a GELU activation function. The first convolutional layer has a 3x3 kernel with stride of 1 and padding of 1. After passing through the subsequent activation function, the second convolutional layer also has a 3x3 kernel with a stride of 2 and padding of 1. Audio features pass through the second GELU function before progressing to the bulk of the audio encoder. A simple diagram is illustrated below.

<p align="center" width="100%">
  <img src="/Images/encoder_stem_diagram.png" width="100%">
</p>

Like any attention-dependent neural network, the relevant embeddings have to understand their relative input position. Whisper uses a fixed sinusoidal positional encoding for the audio encoder. The stem output is summed with the pertinent position information before progressing to the bulk of the encoder block, seen below. Identical to the original Transformer architecture, these encoder blocks contain two sub-layers. A residual connection exists around each sub-layer, represented in the diagram below as the green arrows below the operational blocks. In the first sub-layer, audio features are immediately normalized before multi-head self-attention is applied. The second sub-layer is a feed-forward network, with two linear projections sandwiching a GELU activation function. The output of each encoder block goes to two locations. First, it feeds into the next block, serving as the adjacent encoder block's input. Second, it serves as the key and value vectors for the cross-attention mechanism in the corresponding decoding block. More details on self- and cross-attention will be provided later.

<p align="center" width="100%">
  <img src="/Images/encoder_diagram.png" width="100%">
</p>

Stacking the blocks described above determines the model complexity. Larger, more complex models lead to better performance. Interestingly,  researchers were able to create a competent model with only 4 encoder and decoder blcoks each, fewer than the original Transformer architecture which contained 6. Researchers observed the best-performing model required 32 encoding and decoding blocks, but offered optimal transcription, translation, and language recognition. The different model sizes can be seen below. 

<p align="center" width="100%">
  <img src="/Images/model_sizes.png" width="50%">
</p>

## Decoder

Describe decoder architecture (embedding and encodings), sublayers (layernorm, MHA, residual connection --> layernorm, cross-attention with encoder output, residual connection --> layernorm, FFN, residual connection). Mention that the base model has the same number of blocks as the original Attention Is All You Need paper. Additionally, Whisper has tiny model which has fewer blocks (4) than Attention Is All You Need paper. 

Maybe something about how as the model gets bigger, hallucination becomes a potential issue, similar to LLMs.

## Attention

Reference to Transformer model origin being a result of Is Attention All You Need paper. Describe attention, why it was created (quantifying comaptibility value between query and key vector). How it's implemented (encoders and decoders interacting, query, keys, values, scaled dot-product attention, linear projections, masking for autoregressive generation). 

Could venture into beam search decoding here. 

# Engineering

## Attention to Detail

## Optimizations
