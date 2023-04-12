import tensorflow as tf
import sys

class BaseFullyConnectedNet(tf.keras.Model):
    """ Generator network.
    """
    def __init__(self, input_dim, output_dim, model_name, nb_units=[256, 256, 256], batchnorm=False):  
        super(BaseFullyConnectedNet, self).__init__()
        self.input_layer = tf.keras.layers.Input((input_dim,))
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model_name = model_name
        self.nb_units = nb_units
        self.batchnorm = batchnorm
        self.all_layers = []
        """ Builds the FC stacks. """
        for i in range(len(nb_units) + 1):
            units = self.output_dim if i == len(nb_units) else self.nb_units[i]
            fc_layer = tf.keras.layers.Dense(
                units = units,
                activation = None,
                kernel_regularizer = tf.keras.regularizers.L2(2.5e-5)
            )   
            norm_layer = tf.keras.layers.BatchNormalization()
            self.all_layers.append([fc_layer, norm_layer])
        
        self.out = self.call(self.input_layer)

    def call(self, inputs, training=True):
        """ Return the output of the Generator.
        Args:
            inputs: tensor with shape [batch_size, input_dim]
        Returns:
            Output of Generator.
            float32 tensor with shape [batch_size, output_dim]
        """
        for i, layers in enumerate(self.all_layers[:-1]):
            # Run inputs through the sublayers.
            fc_layer, norm_layer = layers
            with tf.name_scope("%s_layer_%d" % (self.model_name, i+1)):
                x = fc_layer(inputs) if i==0 else fc_layer(x)
                if self.batchnorm:
                    x = norm_layer(x)
                x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        fc_layer, norm_layer = self.all_layers[-1]
        with tf.name_scope("%s_layer_ouput" % self.model_name):
            output = fc_layer(x)
            # No activation func at last layer
            #x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        return output

class Discriminator(tf.keras.Model):
    """Discriminator network.
    """
    def __init__(self, input_dim, model_name, nb_units=[256, 256], batchnorm=True):  
        super(Discriminator, self).__init__()
        self.input_layer = tf.keras.layers.Input((input_dim,))
        self.input_dim = input_dim
        self.model_name = model_name
        self.nb_units = nb_units
        self.batchnorm = batchnorm
        self.all_layers = []
        """Builds the FC stacks."""
        for i in range(len(self.nb_units)+1):
            units = 1 if i == len(self.nb_units) else self.nb_units[i]
            fc_layer = tf.keras.layers.Dense(
                units = units,
                activation = None
            )
            norm_layer = tf.keras.layers.BatchNormalization()

            self.all_layers.append([fc_layer, norm_layer])
        self.out = self.call(self.input_layer)

    def call(self, inputs, training=True):
        """Return the output of the Discriminator network.
        Args:
            inputs: tensor with shape [batch_size, input_dim]
        Returns:
            Output of Discriminator.
            float32 tensor with shape [batch_size, 1]
        """
            
        for i, layers in enumerate(self.all_layers[:-1]):
            # Run inputs through the sublayers.
            fc_layer, norm_layer = layers
            with tf.name_scope("%s_layer_%d" % (self.model_name,i+1)):
                x = fc_layer(inputs) if i==0 else fc_layer(x)
                if self.batchnorm:
                    x = norm_layer(x)
                x = tf.keras.activations.tanh(x)
                #x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        fc_layer, norm_layer = self.all_layers[-1]
        with tf.name_scope("%s_layer_ouput" % self.model_name):
            output = fc_layer(x)
        return output

class Discriminator_img(tf.keras.Model):
    """Discriminator network for Mnist dataset.
    """
    def __init__(self, input_dim, model_name, nb_units=[32, 64], batchnorm=True, dataset='mnist'):  
        super(Discriminator_img, self).__init__()
        self.nb_channels = 1 if dataset=='mnist' else 3
        self.input_layer = tf.keras.layers.Input((input_dim,input_dim,self.nb_channels))
        self.input_dim = input_dim
        self.model_name = model_name
        self.nb_units = nb_units
        self.batchnorm = batchnorm
        self.all_layers = []
        """Builds the Conv stacks."""
        if dataset=='mnist':
            self.conv1_layer =  tf.keras.layers.Conv2D(self.nb_units[0], (5, 5), strides=(2, 2), padding='same',
                                        input_shape=[28,28,1],activation='relu')
        elif dataset=='cifar10':
            self.conv1_layer =  tf.keras.layers.Conv2D(self.nb_units[0], (5, 5), strides=(2, 2), padding='same',
                                        input_shape=[32,32,3],activation='relu')
        self.dropout1_layer = tf.keras.layers.Dropout(0.3)
        self.conv2_layer =  tf.keras.layers.Conv2D(self.nb_units[1], (5, 5), strides=(2, 2), padding='same',
                                    activation='relu')    
        self.dropout2_layer = tf.keras.layers.Dropout(0.3)
        self.dense_layer = tf.keras.layers.Flatten()
        self.out_layer = tf.keras.layers.Dense(1)
        self.out = self.call(self.input_layer)

    def call(self, inputs, training=True):
        """Return the output of the Discriminator network.
        Args:
            inputs: tensor with shape [batch_size, input_dim]
        Returns:
            Output of Discriminator.
            float32 tensor with shape [batch_size, 1]
        """
        x = self.conv1_layer(inputs)
        x = self.dropout1_layer(x)
        x = self.conv2_layer(x)
        x = self.dropout2_layer(x)
        x = self.dense_layer(x)
        with tf.name_scope("%s_layer_ouput" % self.model_name):
            output = self.out_layer(x)
        return output


class Covolution2DNet(tf.keras.Model):
    """ Mnist image encoder (None, 28, 28, 1) to (None, output_dim)
    Cifar10 image encoder (None, 32, 32, 3) to (None, output_dim)
    """
    def __init__(self, input_dim, output_dim, model_name, nb_units=[32, 64], batchnorm=False):  
        super(Covolution2DNet, self).__init__()
        self.nb_channels = 1 if input_dim==28 else 3
        self.input_layer = tf.keras.layers.Input((input_dim,input_dim,self.nb_channels))
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model_name = model_name
        self.nb_units = nb_units
        self.batchnorm = batchnorm
        self.all_layers = []
        """ Builds the Conv stacks. """
        self.conv1_layer = tf.keras.layers.Conv2D(
                filters=self.nb_units[0], kernel_size=3, strides=(2, 2), activation='relu')
        self.conv2_layer = tf.keras.layers.Conv2D(
                filters=self.nb_units[1], kernel_size=3, strides=(2, 2), activation='relu')
        self.flatten_layer = tf.keras.layers.Flatten()
        self.dense_layer = tf.keras.layers.Dense(units = output_dim)
        self.out = self.call(self.input_layer)

    def call(self, inputs, training=True):
        """ Return the output of the Encoder.
        Args:
            inputs: tensor with shape [batch_size, input_dim, input_dim, 1]
        Returns:
            Output of Generator.
            float32 tensor with shape [batch_size, output_dim]
        """
        x = self.conv1_layer(inputs)
        x = self.conv2_layer(x)
        x = self.flatten_layer(x)
        with tf.name_scope("%s_layer_ouput" % self.model_name):
            output = self.dense_layer(x)
            #x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        return output


class Covolution2DTransposedNet(tf.keras.Model):
    """ Mnist image encoder (None, input_dim) to (None, 28, 28, 1) 
    """
    def __init__(self, input_dim, model_name, nb_units=[64, 32], batchnorm=False, dataset='mnist'):  
        super(Covolution2DTransposedNet, self).__init__()
        self.input_layer = tf.keras.layers.Input((input_dim,))
        self.input_dim = input_dim
        self.nb_channels = 1 if dataset=='mnist' else 3
        self.model_name = model_name
        self.nb_units = nb_units
        self.batchnorm = batchnorm
        self.all_layers = []
        """ Builds the Conv stacks. """
        if dataset=='mnist':
            self.dense_layer = tf.keras.layers.Dense(units = 7*7*32)
            self.reshape_layer = tf.keras.layers.Reshape(target_shape=(7, 7, 32))
        elif dataset=='cifar10':
            self.dense_layer = tf.keras.layers.Dense(units = 8*8*32)
            self.reshape_layer = tf.keras.layers.Reshape(target_shape=(8, 8, 32))   
        else:
            print('Error dataset')
            sys.exit()
        self.logvar_dense_layer = tf.keras.layers.Dense(units = 1)         
        self.conv1trans_layer = tf.keras.layers.Conv2DTranspose(
                filters=self.nb_units[0], kernel_size=3, strides=2, padding='same',
                activation='relu')
        self.conv2trans_layer = tf.keras.layers.Conv2DTranspose(
                filters=self.nb_units[1], kernel_size=3, strides=2, padding='same',
                activation='relu')
        self.conv3trans_layer = tf.keras.layers.Conv2DTranspose(
                filters=self.nb_channels, kernel_size=3, strides=1, padding='same')
        self.out = self.call(self.input_layer)

    def call(self, inputs, training=True):
        """ Return the output of the Encoder.
        Args:
            inputs: tensor with shape [batch_size, input_dim]
        Returns:
            Output of Generator.
            float32 tensor with shape [batch_size, 28, 28, 1], [batch, 1]
        """
        x = self.dense_layer(inputs)
        logvar = self.logvar_dense_layer(x)
        x = self.reshape_layer(x)
        x = self.conv1trans_layer(x)
        x = self.conv2trans_layer(x)
        with tf.name_scope("%s_layer_ouput" % self.model_name):
            output = self.conv3trans_layer(x)
            #x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        return output,logvar