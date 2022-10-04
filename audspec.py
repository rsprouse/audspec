#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 14:19:18 2017

@author: ubuntu
"""
import numpy as np
import librosa
from scipy.fft import rfft
from numpy.polynomial import Polynomial

class Audspec(object):
    '''
    An Audspec object is an instantiation of analysis parameters and routines
    for creating auditory spectrograms.
    '''
    def __init__(self, fs, step_size, maxcbfiltn):
        super(Audspec, self).__init__()
        self.fs = np.float32(fs)
        self.dft_n = 2**(np.int32(np.log2(0.05*fs)))  # choose fft size based on fs
        spect_points = int(self.dft_n/2) + 1
        self.spect = np.zeros(spect_points)
        self.spect_times = np.zeros(0)
        self.inc = self.fs/self.dft_n;		# get hz stepsize in fft

        self.topbark = self.hz2bark(self.fs/2.0)
        self.ncoef = np.int32(self.topbark * 3.5)  # number of points in the auditory spectrum
        self.zinc = self.topbark/(self.ncoef+1)
        self.fft_freqs = np.arange(1, spect_points+1) * self.inc
        self.zfreqs = np.arange(1, self.ncoef+1) * self.zinc
        self.freqs = self.bark2hz(self.zfreqs)

        self.step_size = np.float32(step_size)  # temporal intervals in sec
        self.maxcbfiltn = np.int32(maxcbfiltn)  # number of points in biggest CB filter

        self.cbfilts = self.make_cb_filters().astype(np.float32)
        self.window = np.hamming(self.dft_n)

        # TODO: add citation for these polynomial coefficients - Moore & Glasberg, 1987?
        loudp = Polynomial([2661.8, -3690.6, 1917.4, -440.77, 37.706])
        self.loud = 10.0**( loudp(np.log10(self.fft_freqs)) / 10.0 ).astype(np.float32)

    def hz2bark(self, hz):
        '''
        Convert frequency in Hz to Bark using the Schroeder 1977 formula.

        Parameters
        ----------

        hz: scalar or array
        Frequency in Hz.

        Returns
        -------
        scalar or array
        Frequency in Bark, in an array the same size as `hz`, or a scalar if
        `hz` is a scalar.
        '''
        return 7 * np.arcsinh(hz/650)

    def bark2hz(self, bark):
        '''
        Convert frequency in Hz to Bark using the Schroeder 1977 formula.

        Parameters
        ----------

        bark: scalar or array
        Frequency in Bark.

        Returns
        -------
        scalar or array
        Frequency in Hz, in an array the same size as `bark`, or a scalar if
        `bark` is a scalar.
        '''
        return 650 * np.sinh(bark/7)

    def make_cb_filters(self):
        '''
        Create and return 2d array of Patterson filters for DFT spectra based
        on attribute values in `self`.

        Patterson, R.D. (1976) Auditory filter shapes derived with noise stimuli. 
                               JASA 59, 640-54.

        The returned filters are stored in an 2d array in which the
        rows represent the filter frequency bands in ascending order. The
        columns contain symmetrical filter coefficients as determined by the
        Patterson formula and centered at the filter frequency in the
        DFT spectrum. Filter coefficients outside the frequency band are set
        to 0.0. To view the filter coefficients for band `j` do
        `self.cbfilts[j][self.cbfilts[j] != 0.0]`.

        The one-sided length of filter coefficients for each band is stored
        in the `cbfiltn` attribute. The number of coefficients in the
        symmetrical filter for band `j` is therefore
        `(self.cbfiltn[j] * 2) - 1`. In a few bands this calculation might not
        be correct since the coefficients may not fit when the center frequency
        is near the left or right edge of the DFT spectrum. In such cases the
        coefficients are truncated, and the actual number of coefficients for
        the band `j` can be found with `np.sum(self.cbfilts[j] != 0.0)`.
        '''
        cbfilts = np.zeros([len(self.freqs), len(self.spect)])
        dfreq = np.arange(self.maxcbfiltn) * self.inc
        cbfiltn = np.searchsorted(dfreq, self.freqs / 5)
        cbfiltn[cbfiltn > self.maxcbfiltn] = self.maxcbfiltn
        self.cbfiltn = cbfiltn
        for j, iidx in enumerate(cbfiltn):
            cbfilt = np.zeros(self.maxcbfiltn)
            bw = 10.0 ** ( (8.3 * np.log10(self.freqs[j]) - 2.3) / 10.0 )
            hsq = np.exp(-np.pi * ((dfreq[:iidx] / bw) ** 2))
            cbfilt[:iidx] = np.sqrt(hsq)
            cbfilt /= cbfilt[0] + np.sum(cbfilt[1:] * 2)

            # Make a symmetrical array of coefficients centered at loc.
            # [n, n-1, ..., 2, 1, 0, 1, 2, ... n-1, n]
            loc = (self.freqs[j] / self.inc).astype(int) # get location in dft spectrum
            left_n = iidx if iidx <= loc else loc
            right_n = iidx if loc + iidx < (self.dft_n / 2) else int(self.dft_n / 2) - loc
            coef = np.append(np.flip(cbfilt[:left_n])[:-1], cbfilt[:right_n])
            startidx = loc - left_n + 1
            endidx = loc + right_n
            cbfilts[j, startidx:endidx] = coef
        return cbfilts

    def create_sharp_filter(self, span=4, mult=3, dimension = "frequency"):
        '''
        Create and return a 1d sharpening filter symmetric in frequency, or time

        Parameters
        ----------

        span: scalar

        mult: scalar

        Returns
        -------
        1d sharpening filter
        '''
        
        if (dimension=="frequency"):  # default value
            steps = np.int32(span / self.zinc)
        else:  # otherwise assume temporal sharpening
            steps = np.int32(span / self.step_size)
            
        if steps % 2 == 0:
            steps += 1
        sharp = np.full(steps, -1.0)
        mid = int(steps / 2)
        sharp[mid] = steps * mult
        sharp /= sharp.sum()
        return sharp 
     

    def create_blur_filter(self, span=3, sigma=1.5):
        '''
        Create and return a 1d Gaussian blur filter.

        Parameters
        ----------

        span: scalar

        sigma: scalar

        Returns
        -------
        1d blur filter
        '''
        steps = np.int32(span / self.zinc)
        if steps % 2 == 0:
            steps += 1
        mid = int(steps / 2)
        blur = 1 / (np.sqrt(np.pi*2) * sigma) * \
            np.exp(((np.arange(-mid, mid+1) ** 2) * -1) / (2 * sigma**2))
        blur /= blur.sum()
        return blur

    def apply_filt(self, gram, filt, axis, half_rectify):
        '''
        Apply a filter along the axis of an auditory spectrogram. Spectrogram
        values are also rescaled after filtering.

        Parameters
        ----------

        gram: 2d array
        Auditory spectrogram.

        filt: 1d array
        A filter.

        axis: 0 or 1
        The axis to iterate over and apply the filter.

        half_rectify: bool
        If True, apply half-wave rectification to filtered spectrogram.

        Returns
        -------
        2d array
        The auditory spectrogram after the filter has been applied. The shape
        is the same as `gram`.
        '''
        # Make the axis to act on the first dimension if required.
        if axis == 1:
            gram = gram.transpose()

        # Do convolution along the first dimension.
        agram = np.zeros_like(gram)
        mid = (len(filt) - 1) / 2
        for j in np.arange(gram.shape[0]):
            agram[j] = np.convolve(
                np.pad(gram[j], int(mid), mode='edge'),
                filt,
                mode='valid'
            )

        # Do half-wave rectification if requested.
        if half_rectify is True:
            agram[agram < 0] = 0

        # Rescale spectrogram values as a ratio of the distance of each
        # value from the minimum to the range of values in the spectrogram.
        mymin = np.min(agram)
        mymax = np.max(agram)
        agram = (agram - mymin) / (mymax - mymin)

        if axis == 1:
            return agram.transpose()
        else:
            return agram

        
    def make_spect(self, data, *args, **kwargs):
        '''
        Make an acoustic spectrogram via rfft().

        Parameters
        ----------

        data: 1d array
        Audio data.

        kwargs: dict, optional
        Keyword arguments will be passed to the scipy.fft.rfft() function.

        Returns
        -------
        None. (The 2d acoustic spectrogram is added to self.spect.)
        '''
        data = np.pad(data, [int(self.dft_n/2), int(self.dft_n/2)], 'constant')
        if (np.max(data) < 1):   # floating point but we need 16bit int range
            data = (data*(2**15)) #.astype(np.int32)

        print(f'padded data, dtype is {data.dtype}')
        hop = int(self.step_size * self.fs)
        print(f'making frames of length {self.dft_n} and step {hop}')
        frames = librosa.util.frame(data, frame_length=self.dft_n, hop_length=hop).transpose()
        self.spect_times = librosa.frames_to_time(
            np.arange(frames.shape[0]),
            sr=self.fs,
            hop_length=hop,
            n_fft=self.dft_n
        )
        print(f'made frames, shape {frames.shape}')
        # Add some noise, then scale frames by the window.
        frames = (frames + np.random.normal(0, 1, frames.shape)) * self.window
        print('starting rfft')
        A = rfft(frames, **kwargs)
        print(f'created A, shape {A.shape}')
        self.spect = (np.abs(A) - self.loud).astype(np.float32)
        print(f'created spect, shape {self.spect.shape}')
        self.spect[self.spect < 1] = 1
        print('spect non-negative')
        dur = data.shape[0] / self.fs
        half_actual_step = hop / self.fs / 2
        self.spect_times_linspace = np.linspace(half_actual_step, dur - half_actual_step, self.spect.shape[0])
        return

    def make_zgram(self):
        '''
        Make an auditory spectrogram by applying the Patterson filters to the
        acoustic spectrogram.

        Parameters
        ----------

        spect: 2d array
        Acoustic spectrogram.

        Returns
        -------
        The 2d auditory spectrogram.
        '''
        print('adding axis')
        print(f'self.spect shape {self.spect.shape}')
        print(f'cbfilts shape {self.cbfilts.shape}')
        zgram = self.spect[:, np.newaxis, :] * self.cbfilts[np.newaxis, :, :]
        print('summing')
        zgram = zgram.sum(axis=2)
        print('log10')
        zgram = 10 * np.log10(zgram)
        print('returning')
        return zgram