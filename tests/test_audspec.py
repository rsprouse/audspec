#!/usr/bin/env python

import librosa
import pytest
import numpy as np
from pathlib import Path
import audspec

resdir = Path('tests')
fs = 22050

@pytest.fixture
def sine377file():
    return resdir / 'sineWithNoise.377hz.wav'

@pytest.fixture
def audspec_defaults():
    return {
        'step': 0.005,   # The frame step for the spectrogram, in seconds.
        'maxpatterson_coef': 100
    }

@pytest.fixture
def rfft_defaults():
    # Params passed to rfft() in make_zgram()
    return {
        'overwrite_x': True,  # Try to reduce memory usage
        'workers': -1         # Use all CPUs
    }

@pytest.fixture
def sine377spec(sine377file, audspec_defaults, rfft_defaults):
    data, _ = librosa.load(sine377file, sr=fs)
    aud = audspec.Audspec(
        fs,
        step_size=audspec_defaults['step'],
        maxcbfiltn=audspec_defaults['maxpatterson_coef']
    )
    # Create acoustic spectrogram.
    (spect, spect_times, spect_times_linspace) = aud._make_spect(data, rfft_defaults)
    return (spect, spect_times, spect_times_linspace)

@pytest.fixture
def sine377(sine377file, audspec_defaults, rfft_defaults):
    aud = audspec.Audspec(
        fs,
        step_size=audspec_defaults['step'],
        maxcbfiltn=audspec_defaults['maxpatterson_coef']
    )
    # Create auditory spectrogram.
    aud.make_zgram(sine377file, rfft_defaults)
    return aud

@pytest.fixture
def sine377_with_5_1(sine377):
    '''
    Set all values of the first five spectral slices of the zgram to 1.
    '''
    sine377.zgram[:5,:] = 1
    return sine377

@pytest.fixture
def sine377_with_5_1_igram_lgram_ogram(sine377_with_5_1):
    '''
    Create sample filtered zgrams and return as a dict.
    '''
    # Create filters.
    sharp_1 = sine377_with_5_1.create_sharp_filter(span=3, mult=2)
    blur = sine377_with_5_1.create_blur_filter(span=3, sigma=3)
    sharp_2 = sine377_with_5_1.create_sharp_filter(span=6, mult=1)
    temporal_sharp = sine377_with_5_1.create_sharp_filter(
        span=0.05, mult=1, dimension="time"
    )

    # Apply filters.
    lgram = sine377_with_5_1.apply_filt(
        sine377_with_5_1.zgram, sharp_1, axis=0, half_rectify=True
    )
    igram = sine377_with_5_1.apply_filt(
        lgram, blur, axis=0, half_rectify=False
    )
    lgram = sine377_with_5_1.apply_filt(
        lgram, sharp_2, axis=0, half_rectify=True
    )
    ogram = sine377_with_5_1.apply_filt(
        sine377_with_5_1.zgram, temporal_sharp, axis=1, half_rectify=True
    )
    return (
        sine377_with_5_1,
        {
            'igram': igram,
            'lgram': lgram,
            'ogram': ogram,
        }
    )
    
def test_wav_load():
    '''
    Test that we can load .wav files.
    '''
    data, rate = librosa.load(resdir / 'sineWithNoise.377hz.wav', sr=fs)

def test_zgram_create_spec(sine377spec):
    '''
    '''
    spect, spect_times, spect_times_linspace = sine377spec
    assert(spect.shape[0] == len(spect_times))

def test_zgram_create_zgram_from_data(sine377file, audspec_defaults, rfft_defaults):
    data, _ = librosa.load(sine377file, sr=fs)
    aud = audspec.Audspec(
        fs,
        step_size=audspec_defaults['step'],
        maxcbfiltn=audspec_defaults['maxpatterson_coef']
    )
    # Create auditory spectrogram.
    aud.make_zgram(sine377file, rfft_defaults)
    assert(aud.spect.shape == (len(aud.spect_times), len(aud.fft_freqs)))
    assert(aud.zgram.shape == (len(aud.spect_times), len(aud.freqs)))

def test_zgram_create_zgram_from_path(sine377file, audspec_defaults, rfft_defaults):
    aud = audspec.Audspec(
        fs,
        step_size=audspec_defaults['step'],
        maxcbfiltn=audspec_defaults['maxpatterson_coef']
    )
    # Create auditory spectrogram.
    aud.make_zgram(sine377file, rfft_defaults)
    assert(aud.spect.shape == (len(aud.spect_times), len(aud.fft_freqs)))
    assert(aud.zgram.shape == (len(aud.spect_times), len(aud.freqs)))

def test_zgram_shape(sine377):
    '''
    Check that the zgram axes are the correct orientation and length. When
    comparing to the `freqs` property we should find that the max of each
    spectral slice in the zgram is near the 377Hz center frequency that was
    used in the formula that generated the input sound.
    '''
    # Shape should be (times, freqs).
    assert(
        np.all(
            sine377.zgram.shape == (
                len(sine377.spect_times),
                len(sine377.freqs)
            )
        )
    )
    assert(
        np.all(
            np.argmax(sine377.zgram, axis=1) == \
            np.argmin(np.abs(sine377.freqs - 377))
        )
    )

def test_zgram_igram_lgram_ogram(sine377_with_5_1_igram_lgram_ogram):
    aud = sine377_with_5_1_igram_lgram_ogram[0]
    layers = sine377_with_5_1_igram_lgram_ogram[1]
    for k in layers.keys():
        assert(np.all(layers[k].shape == aud.zgram.shape))

def test_zgram_save(sine377_with_5_1_igram_lgram_ogram):
    aud = sine377_with_5_1_igram_lgram_ogram[0]
    filtered_imgs = sine377_with_5_1_igram_lgram_ogram[1]
    assert(aud.spect.shape[0] == aud.zgram.shape[0])
    assert(aud.spect.shape[0] == len(aud.spect_times))
    assert(aud.spect.shape[1] == len(aud.fft_freqs))
    assert(aud.zgram.shape[1] == len(aud.freqs))
    aud.savez('test.npz', layers=filtered_imgs)

def test_zgram_read():
    '''
    Test reading of the .npz file created by `test_zgram_save`.
    '''
    d = np.load('test.npz', mmap_mode='r')
    for k in d.files:
        print(f'file {k} in archive')
    assert(d['fs'] == fs)
    assert(np.all(d['layer_names'] == ('igram', 'lgram', 'ogram')))
    assert(d['spect'].shape[0] == d['zgram'].shape[0])
    assert(d['spect'].shape[0] == len(d['spect_times']))
    assert(d['spect'].shape[1] == len(d['fft_freqs']))
    assert(d['zgram'].shape[1] == len(d['freqs']))
    assert(
        np.all(
            d['ogram'].shape == (
                len(d['spect_times']),
                len(d['freqs'])
            )
        )
    )
    # The first five spectral slices were assigned the value 1 in 
    # `test_zgram_save`.
    assert(np.all(d['zgram'][:5,:] == 1))
