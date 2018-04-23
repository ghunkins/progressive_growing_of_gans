import pickle
import numpy as np
import tensorflow as tf
import PIL.Image
import sys
import interpolate

# add path
sys.path.append('.')

# Initialize TensorFlow session.
tf.InteractiveSession()

# Import official CelebA-HQ networks.
with open('karras2018iclr-celebahq-1024x1024.pkl', 'rb') as file:
    G, D, Gs = pickle.load(file)

# Generate latent vectors.
latents = np.random.RandomState(1000).randn(1000, *Gs.input_shapes[0][1:]) # 1000 random latents
print('Latents #1 Size:', latents.shape)
#latents = latents[[477, 56, 83, 887, 583, 391, 86, 340, 341, 415]] # hand-picked top-10
chosen_latents = latents[[83, 887, 583]]
latents = []
for idx, ratio in enumerate(np.linspace(0, 1, 10)):
	z1 = np.stack([interpolate.slerp(ratio, r1, r2) for r1, r2 in zip(chosen_latents[0], chosen_latents[1])])
	z2 = np.stack([interpolate.slerp(ratio, r1, r2) for r1, r2 in zip(chosen_latents[0], chosen_latents[2])])
	z3 = np.stack([interpolate.slerp(ratio, r1, r2) for r1, r2 in zip(chosen_latents[1], chosen_latents[2])])
	latents.extend([z1, z2, z3])

latents = np.array(latents)
print('Latents #2 Size:', latents.shape)

# Generate dummy labels (not used by the official networks).
labels = np.zeros([latents.shape[0]] + Gs.input_shapes[1][1:])
print('Labels Size:', labels.shape)

# Run the generator to produce a set of images.
images = Gs.run(latents, labels)

print('Images #1 Size:',images.shape)

# Convert images to PIL-compatible format.
images = np.clip(np.rint((images + 1.0) / 2.0 * 255.0), 0.0, 255.0).astype(np.uint8) # [-1,1] => [0,255]
images = images.transpose(0, 2, 3, 1) # NCHW => NHWC

print('Images #2 Size:',images.shape)

# Save images as PNG.
for idx in range(images.shape[0]):
    PIL.Image.fromarray(images[idx], 'RGB').save('img%d.png' % idx)