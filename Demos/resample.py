import matplotlib.pyplot as plt
from optbayesexpt import ParticlePDF
import numpy as np

rng = np.random.default_rng()
plt.rc('font', size=13)

n_samples = 100
x_samples = rng.normal(0, 1, n_samples)
pdf = ParticlePDF((x_samples,))

coords = pdf.particles

fig, subplots = plt.subplots(nrows=3, ncols=1, sharex=True)

plt.sca(subplots[0])
plt.stem(coords[0], pdf.particle_weights, basefmt='',
         use_line_collection=True)

plt.text(.01, .8, "(a) Samples of normal distribution, "
                  "uniform weight", transform=subplots[0].transAxes)
plt.xlim(-3, 3)
plt.ylim(0, .015)
plt.ylabel("weights")

plt.sca(subplots[1])
x_samples = rng.uniform(-5, 5, n_samples)
pdf2 = ParticlePDF((x_samples,))
weight = np.exp(-1 * x_samples ** 2 / 2) / np.sqrt(2 * np.pi)
pdf2.particle_weights = weight / np.sum(weight)
plt.stem(pdf2.particles[0], pdf2.particle_weights, basefmt='',
         use_line_collection=True)

plt.text(.01, .8, "(b) Samples of uniform distribution, "
                  "normal weight", transform=subplots[1].transAxes)
plt.ylim(0, .06)
plt.ylabel("weights")

plt.sca(subplots[2])
pdf2.resample()
plt.stem(pdf2.particles[0], pdf2.particle_weights, basefmt='',
         use_line_collection=True)

plt.ylabel("weights")
plt.ylim(0, .015)
plt.text(.01, .8, "(c) Resampled version of (b)",
         transform=subplots[2].transAxes)
plt.xlabel("parameter sample value")

plt.tight_layout()
plt.show()
