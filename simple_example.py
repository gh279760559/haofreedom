import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# %% plot stuff
def f(t):
    return np.exp(-t) * np.cos(2*np.pi*t)


t1 = np.arange(0.0, 5.0, 0.1)
t2 = np.arange(0.0, 5.0, 0.02)

plt.figure(1)
plt.subplot(211)
plt.plot(t1, f(t1), 'bo', t2, f(t2), 'k')

plt.subplot(212)
plt.plot(t2, np.cos(2*np.pi*t2), 'r--')
plt.show()

# %% a
img = mpimg.imread('messi5.jpg')

imgplot = plt.imshow(img)

lum_img = img[:, :, 0]

plt.imshow(lum_img)

plt.plot(range(100))
plt.show()

# %% basic learn
a = 3

# %% function learn


def create_adder(x):
    def adder(y):
        return x + y
    return adder


add_10 = create_adder(10)
add_10(3)  # => 13
add_10(12)
