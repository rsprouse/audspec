# audspec

Auditory spectrogram creation from single-channel audio files.

## Installation

To install:

```bash
pip install git+https://github.com/rsprouse/audspec
```

Or

```bash
git clone https://github.com/rsprouse/audspec
cd audspec
python setup.py install
```

Run tests with:

```bash
python -m pytest    # From the source root directory
```

## Synopsis

To create and cache an auditory spectrogram:

```python
from audspec import Audspec

aud = Audspec(fs, step_size=0.005, maxcbfiltn=100)
aud.make_zgram('myaudio.wav')
aud.savez('myaudio.audspec.npz')
```

To read the cached auditory spectrogeram:

```python
d = np.load('myaudio.audspec.npz', mmap_mode='r')
zgram = d['zgram']
```

For more detailed usage see the notebooks in the [`doc`](doc) directory.

## Authors

Keith Johnson (keithjohnson@berkeley.edu) is the primary author.

Minor improvements and documentation notebooks by Ronald L. Sprouse (ronald@berkeley.edu).
