import tensorflow as tf

def vgg_block(inputs, filters, kernel_size, name, data_format, training=False,
              batch_normalization=True, kernel_reg=0., **params):
    with tf.compat.v1.variable_scope(name, reuse=tf.compat.v1.AUTO_REUSE):
        x = tf.keras.layers.Conv2D(filters, kernel_size, name='conv',
                       kernel_regularizer=tf.keras.regularizers.l2(0.5 * (kernel_reg)),
                       data_format=data_format, **params)(inputs)
        if batch_normalization:
            x = tf.keras.layers.BatchNormalization(
                    name='bn', fused=True,
                    axis=1 if data_format == 'channels_first' else -1)(x, training=training)
    return x


def vgg_backbone(inputs, train, **config):
    params_conv = {'padding': 'SAME', 'data_format': config['data_format'],
                   'activation': tf.nn.relu, 'batch_normalization': True,
                   'training': train,
                   'kernel_reg': config.get('kernel_reg', 0.)}
    params_pool = {'padding': 'SAME', 'data_format': config['data_format']}

    with tf.compat.v1.variable_scope('vgg', reuse=tf.compat.v1.AUTO_REUSE):
        x = vgg_block(inputs, 64, 3, 'conv1_1', **params_conv)
        x = vgg_block(x, 64, 3, 'conv1_2', **params_conv)
        x = tf.keras.layers.MaxPool2D(2, 2, name='pool1', **params_pool)(x)

        x = vgg_block(x, 64, 3, 'conv2_1', **params_conv)
        x = vgg_block(x, 64, 3, 'conv2_2', **params_conv)
        x = tf.keras.layers.MaxPool2D(2, 2, name='pool2', **params_pool)(x)

        x = vgg_block(x, 128, 3, 'conv3_1', **params_conv)
        x = vgg_block(x, 128, 3, 'conv3_2', **params_conv)
        x = tf.keras.layers.MaxPool2D(2, 2, name='pool3', **params_pool)(x)

        x = vgg_block(x, 128, 3, 'conv4_1', **params_conv)
        x = vgg_block(x, 128, 3, 'conv4_2', **params_conv)

    return x
