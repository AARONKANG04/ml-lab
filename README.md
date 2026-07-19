# ml-lab

A place where I reimplement machine learning papers from scratch to understand
them properly. Reading a paper and being able to build the thing turn out to be
different skills, and this repo is me closing that gap one paper at a time. The
rule I set for myself is to write the core of each model by hand first, not just
call a library, then train it on real data and report honest numbers (including
the parts that didn't work).

## Projects

### [Machine translation](projects/machine-translation)

Sutskever, Vinyals & Le (2014), the LSTM encoder-decoder that showed a plain
neural network could translate without any alignment model. I wrote the LSTM cell
and the encoder/decoder loop by hand, then swapped in cuDNN's `nn.LSTM` to train
at speed on all 40.8M WMT14 English-French pairs. It reaches 14.5 BLEU with beam
search on newstest2013. The project README has the training curves, the full
metrics (BLEU, chrF, TER, ROUGE), and 3,000 example translations with a look at
where it fails.

## How it's set up

Each project is self-contained under `projects/<name>` with its own README,
`requirements.txt`, `config.yaml`, and `src/`. Training runs log to Weights &
Biases and checkpoint to disk so they can be resumed.
