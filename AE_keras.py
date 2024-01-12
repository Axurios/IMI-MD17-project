import keras
import numpy as np

class Autoencoder(keras.Model):

    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()

        self.encoder = keras.Sequential([
            keras.layers.Dense(latent_dim, activation='sigmoid', input_shape=(input_dim,))
        ])

        self.decoder = keras.Sequential([
            keras.layers.Dense(input_dim, activation='sigmoid')
        ])

    def call(self, x):
        z = self.encoder(x)
        y = self.decoder(z)
        return y


def main():
    input_dim = 10
    latent_dim = 5

    autoencoder = Autoencoder(input_dim, latent_dim)

    optimizer = keras.optimizers.Adam(lr=0.001)
    autoencoder.compile(loss='mse', optimizer=optimizer)

    x_train = np.random.rand(100, input_dim)
    autoencoder.fit(x_train, x_train, epochs=100)

    print(autoencoder.predict(x_train))


if __name__ == '__main__':
    main()
