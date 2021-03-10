import tensorflow as tf

from model.transformer_utils import positional_encoding, scaled_dot_product_attention


class CNNResNorm(tf.keras.layers.Layer):
    '''
    custom layer with conv1D + Activations + Normalisation (layer or batch)
    parameterized on types of  activations  (the last can be sigmoid/softmax) 
    when the last is reLu
    '''
    
    def __init__(self,
                 out_size: int,
                 n_layers: int,
                 hidden_size: int,
                 kernel_size: int,
                 inner_activation: str,
                 last_activation: str,
                 padding: str,
                 normalization: str,
                 **kwargs):
        
        super(CNNResNorm, self).__init__(**kwargs)                              # initialize the parent class (=keras.layers.Layer)
        
        
        self.convolutions = [tf.keras.layers.Conv1D(filters=hidden_size,
                                                    kernel_size=kernel_size,
                                                    padding=padding)
                             for _ in range(n_layers - 1)]
        self.inner_activations = [tf.keras.layers.Activation(inner_activation) for _ in range(n_layers - 1)]
        self.last_conv = tf.keras.layers.Conv1D(filters=out_size,
                                                kernel_size=kernel_size,
                                                padding=padding)
        self.last_activation = tf.keras.layers.Activation(last_activation)
        
        if normalization == 'layer':
            self.normalization = [tf.keras.layers.LayerNormalization(epsilon=1e-6) for _ in range(n_layers + 1)]
        elif normalization == 'batch':
            self.normalization = [tf.keras.layers.BatchNormalization() for _ in range(n_layers + 1)]
        else:
            assert False is True, f'normalization must be either "layer" or "batch", not {normalization}.'
    
    def call_convs(self, x, training):
        '''
        helper function used in the function call()
        '''
       
        #1. stack convolutions over n = number of convolutions-1
        for i in range(0, len(self.convolutions)):
            x = self.convolutions[i](x)
            x = self.inner_activations[i](x)
            x = self.normalization[i](x, training=training)
        return x
    
    def call(self, inputs, training):
        
        #2. stack of convolutions
        x = self.call_convs(inputs, training=training) # stack of convolutions
        
        #3. final convolutions
        x = self.last_conv(x)
        x = self.last_activation(x)
        x = self.normalization[-2](x, training=training)
        
        #4. skip/residual
        return self.normalization[-1](inputs + x, training=training)


class FFNResNorm(tf.keras.layers.Layer):
    '''
    custom layer with Dense + Activations + Normalisation (layer only) + DropOut
    all activations are of type 'relu'
    '''
    def __init__(self,
                 model_dim: int,
                 dense_hidden_units: int,
                 dropout_rate: float,
                 **kwargs):
        
        super(FFNResNorm, self).__init__(**kwargs)                       # initialize the parent class (=keras.layers.Layer)
        
        self.d1 = tf.keras.layers.Dense(dense_hidden_units)              # initialize a dense layer parameterized by n=dense_hidden_units
        self.activation = tf.keras.layers.Activation('relu')             # initialize an activation Layer with RELU
        self.d2 = tf.keras.layers.Dense(model_dim)                       # initialize a dense layer parameterized by n=model_dim            
        self.dropout = tf.keras.layers.Dropout(dropout_rate)             # inititalize a dropOut layer
        self.ln = tf.keras.layers.LayerNormalization(epsilon=1e-6)       # initialize a layerNormalisation layer
        self.last_ln = tf.keras.layers.LayerNormalization(epsilon=1e-6)  # initialize a terminal layerNormalisation layer
    
    def call(self, x, training):
        #1. call dense layer d1
        ffn_out = self.d1(x)                               # (batch_size, input_seq_len, model_dim)
        
        #2. call dense layer d2
        ffn_out = self.d2(ffn_out)                         # (batch_size, input_seq_len, model_dim)
        
        #3. call layerNormalisation
        ffn_out = self.ln(ffn_out)                         # (batch_size, input_seq_len, model_dim)
        
        #4. call Activation
        ffn_out = self.activation(ffn_out)                 # (batch_size, input_seq_len, model_dim)
        
        #5. call Dropout
        ffn_out = self.dropout(ffn_out, training=training) # (batch_size, input_seq_len, model_dim)
        
        #6 skip/residual
        return self.last_ln(ffn_out + x)


class HeadDrop(tf.keras.layers.Layer):
    """ Randomly drop n heads. """
    
    def __init__(self, **kwargs):
        super(HeadDrop, self).__init__(**kwargs)                                                # initialize the parent class (=keras.layers.Layer)
    
    def call(self, batch, training: bool, drop_n_heads: int):
        '''
        if not training/drop_n_heads = 0 returns batch
        if training returns batch * head_mask * 1/(switched_on heads)
        '''
        # Check shape format
        if not training or (drop_n_heads == 0):
            return batch
        
        if len(tf.shape(batch)) != 4:                                                           # shape == 4 (batch_size, head_n, 1, 1)
            raise Exception('attention values must be 4 dimensional')
        
        batch_size = tf.shape(batch)[0]                                                         # shape == 4 (batch_size, head_n, 1, 1)
        head_n = tf.shape(batch)[1]
        
        if head_n == 1:
            return batch                                                                        # if 1 head return the batch as it was defined (1 head = default)
        
        # assert drop_n_heads < head_n, 'drop_n_heads must less than number of heads'
        
        #1. creates an array of size (batch_size)
        keep_head_batch = tf.TensorArray(tf.float32, size=batch_size)                          
        
        #2. mask of shape : 111110000 if head_n=9 & drop_n_heads=4
        keep_mask = tf.concat([tf.ones(head_n - drop_n_heads), tf.zeros(drop_n_heads)], axis=0) # 
        
        #3. shuffle the mask  to alternate the switched-off heads
        for i in range(batch_size):
            t = tf.random.shuffle(keep_mask)                                                    
            keep_head_batch = keep_head_batch.write(i, t)                                       # populate the keep_head_batch with the randomized switch off 
                                                                                                # heads for each sample inside 1 batch
        
        #4. after writing elements need to stack (first: write, second: stack)
        keep_head_batch = keep_head_batch.stack() 
        
        #5. # expand of 2 axis  => result : (dima,dimb,1,1)
        keep_head_batch = keep_head_batch[:, :, tf.newaxis, tf.newaxis] 
        
        #6. "multihead batch"
        return batch * keep_head_batch * tf.cast(head_n / (head_n - drop_n_heads), tf.float32)   # multiply the batch by the dropped head mask 
                                                                                                 # and we correct by a factor = tot_head/working heads


class MultiHeadAttention(tf.keras.layers.Layer):
    
    def __init__(self, model_dim: int, num_heads: int, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs).      # import parent class (=keras.layers.Layer)
        
        self.num_heads = num_heads                               # number of heads to perform multi attention
        self.model_dim = model_dim                               # model_dimension
        self.head_drop = HeadDrop()                              # headrop Layer 
        
        assert model_dim % self.num_heads == 0                   # the dimension of the model must be a multiple of the number of heads
                                                                 # each head will see a fraction of model_dim (need for no remainder)
        
        self.depth = model_dim // self.num_heads  # each head will see a fraction of model_dim
        
        self.wq = tf.keras.layers.Dense(model_dim) # dense Layer for the Query (dimension = model_dim)
        self.wk = tf.keras.layers.Dense(model_dim) # dense Layer for the Keys (dimension = model_dim)
        self.wv = tf.keras.layers.Dense(model_dim) # dense Layer for the Values (dimension = model_dim)
        
        self.dense = tf.keras.layers.Dense(model_dim) # dense Layer (dimension = model_dim)
    
    def split_heads(self, x, batch_size: int):
        """ 
        Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        (This function makes sense because model_dim = self.num_heads * self.depth)
        
        """
        
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth)) # "split" the last dimension into num_heads and depth and auto. 
                                                                        # infer the remaining dimension
        return tf.transpose(x, perm=[0, 2, 1, 3])                       # permutation : batch_size, num_heads, seq_len, depth
    
    
    def call(self, v, k, q_in, mask, training, drop_n_heads):
        
        '''
        args : v, q_in, k have the same dimension
        
        '''
        
        
        batch_size = tf.shape(q_in)[0]
        
        #1. define Q,K,V
        q = self.wq(q_in)  # (batch_size, seq_len, model_dim)  # could have been named q to simplify
        k = self.wk(k)  # (batch_size, seq_len, model_dim)
        v = self.wv(v)  # (batch_size, seq_len, model_dim)
        
        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth) # split_heads(tensor, batch_size) c.f above
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth) # split_heads(tensor, batch_size) c.f above
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth) # split_heads(tensor, batch_size) c.f above
        
        
        #2.
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        
        #3.
        scaled_attention = self.head_drop(scaled_attention, training=training, drop_n_heads=drop_n_heads)
        
        #4. transpose scaled attention
        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)
        
        #5. reshape scale attention (bad variable name)
        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.model_dim))  # Reshape into (batch_size, seq_len_q, model_dim)
        
        #6. Attention and the query Q concatenated along last axis
        concat_query = tf.concat([q_in, concat_attention], axis=-1) # concatenate the Attention and the query Q
        
        #7. Dense operation over past output
        output = self.dense(concat_query)  # Dense(model_dim)(concat_query) 
                                           # (batch_size, seq_len_q, model_dim)
        
        return output, attention_weights


class SelfAttentionResNorm(tf.keras.layers.Layer):
    
    def __init__(self,
                 model_dim: int,
                 num_heads: int,
                 dropout_rate: float,
                 **kwargs):
        
        super(SelfAttentionResNorm, self).__init__(**kwargs)                # initialize the parent class (=keras.layers.Layer)
        
        self.mha = MultiHeadAttention(model_dim, num_heads)                 # use the multihead layer
        self.ln = tf.keras.layers.LayerNormalization(epsilon=1e-6)          # Layer Norm with the epsilon for regularization
        self.dropout = tf.keras.layers.Dropout(dropout_rate)                # DropOut for regularization
        self.last_ln = tf.keras.layers.LayerNormalization(epsilon=1e-6)     # Layer Norm with the epsilon for regularization
    
    def call(self, x, training, mask, drop_n_heads):
        
        # 1. call multihead (defined in init above)
        attn_out, attn_weights = self.mha(x, x, x, mask, training=training,
                                          drop_n_heads=drop_n_heads)  # (batch_size, input_seq_len, model_dim)
        
        # 2. layer normalization (defined init above)
        attn_out = self.ln(attn_out)  # (batch_size, input_seq_len, model_dim)
        
        # 3. call dropout (defined init above)
        out = self.dropout(attn_out, training=training)
        
        # 4. residual/skip connection, attention weights (calculated in 1.)
        return self.last_ln(out + x), attn_weights


class SelfAttentionDenseBlock(tf.keras.layers.Layer):
    
    def __init__(self,
                 model_dim: int,
                 num_heads: int,
                 dense_hidden_units: int,
                 dropout_rate: float,
                 **kwargs):
        
        super(SelfAttentionDenseBlock, self).__init__(**kwargs)                             # initialize the parent class (=keras.layers.Layer)
        
        self.sarn = SelfAttentionResNorm(model_dim, num_heads, dropout_rate=dropout_rate).  # initialize a SelfAttentionResNorm layer
        self.ffn = FFNResNorm(model_dim, dense_hidden_units, dropout_rate=dropout_rate)     # initialize a FFNResNorm layer
    
    def call(self, x, training, mask, drop_n_heads):
        
        # 1.call SelfAttentionResNorm  (= multihead + skip + layer norm)
        attn_out, attn_weights = self.sarn(x, mask=mask, training=training, drop_n_heads=drop_n_heads) 
        
        # 2.call FFNResNorm (= fully connected + res + layer_norm)
        return self.ffn(attn_out, training=training), attn_weights


class SelfAttentionConvBlock(tf.keras.layers.Layer):
    
    def __init__(self,
                 model_dim: int,
                 num_heads: int,
                 dropout_rate: float,
                 conv_filters: int,
                 kernel_size: int,
                 conv_activation: str,
                 **kwargs):
        
        super(SelfAttentionConvBlock, self).__init__(**kwargs)                              # initialize the parent class (=keras.layers.Layer)
        
        self.sarn = SelfAttentionResNorm(model_dim, num_heads, dropout_rate=dropout_rate)   # initialize a SelfAttentionResNorm layer
        
        self.conv = CNNResNorm(out_size=model_dim,                                          # initialize CNNResNorm layer
                               n_layers=2,
                               hidden_size=conv_filters,
                               kernel_size=kernel_size,
                               inner_activation=conv_activation,
                               last_activation=conv_activation,
                               padding='same',
                               normalization='batch')
    
    def call(self, x, training, mask, drop_n_heads):
        
        #1. call SelfAttentionResNorm  (= multihead + skip + layer norm)
        attn_out, attn_weights = self.sarn(x, mask=mask, training=training, drop_n_heads=drop_n_heads)
        
        #2. call CNNResNorm (conv + skip/residuals + normalisation)
        conv = self.conv(attn_out)
        
        return conv, attn_weights


class SelfAttentionBlocks(tf.keras.layers.Layer):
    def __init__(self,
                 model_dim: int,
                 feed_forward_dimension: int,
                 num_heads: list,
                 maximum_position_encoding: int,
                 conv_filters: int,
                 dropout_rate: float,
                 dense_blocks: int,
                 kernel_size: int,
                 conv_activation: str,
                 **kwargs):
        
        super(SelfAttentionBlocks, self).__init__(**kwargs)                           # initialize the parent class (=keras.layers.Layer)
        
        self.model_dim = model_dim                                                    # initialize model dimension
        
        self.pos_encoding_scalar = tf.Variable(1.)                                    # position of the last element 
        
        self.pos_encoding = positional_encoding(maximum_position_encoding, model_dim) # positional encoding (triangular encoding)
        
        self.dropout = tf.keras.layers.Dropout(dropout_rate)                          # droput layer for regularization
        
        
        
        # encoder using dense-type attention mechanism
        self.encoder_SADB = [
            SelfAttentionDenseBlock(model_dim=model_dim, dropout_rate=dropout_rate, num_heads=n_heads,
                                    dense_hidden_units=feed_forward_dimension, name=f'{self.name}_SADB_{i}')
            for i, n_heads in enumerate(num_heads[:dense_blocks])] # some heads will be dedicated to dense-type attention: num_heads[:dense_blocks]
        
         # encoder using convolutional-type attention mechanism
        self.encoder_SACB = [
            SelfAttentionConvBlock(model_dim=model_dim, dropout_rate=dropout_rate, num_heads=n_heads,
                                   name=f'{self.name}_SACB_{i}', kernel_size=kernel_size,
                                   conv_activation=conv_activation, conv_filters=conv_filters)
            for i, n_heads in enumerate(num_heads[dense_blocks:])] # some heads will be dedicated to conv-type attention: num_heads[dense_blocks:]
    
    
    
    def call(self, inputs, training, padding_mask, drop_n_heads, reduction_factor=1):
        
        #1. store the length
        seq_len = tf.shape(inputs)[1]
        
        #2. initialize by multiplying by np.sqrt(model_dim)
        x = inputs * tf.math.sqrt(tf.cast(self.model_dim, tf.float32))
        
        #3. reduction_factor=?
        x += self.pos_encoding_scalar * self.pos_encoding[:, :seq_len * reduction_factor:reduction_factor, :]
        x = self.dropout(x, training=training)
        
        #4.
        attention_weights = {}
        
        #5. transform x by the dense-type attention block
        for i, block in enumerate(self.encoder_SADB):
            x, attn_weights = block(x, training=training, mask=padding_mask, drop_n_heads=drop_n_heads)
            attention_weights[f'{self.name}_DenseBlock{i + 1}_SelfAttention'] = attn_weights
            
        #6. transform x by the conv-type attention block
        for i, block in enumerate(self.encoder_SACB):
            x, attn_weights = block(x, training=training, mask=padding_mask, drop_n_heads=drop_n_heads)
            attention_weights[f'{self.name}_ConvBlock{i + 1}_SelfAttention'] = attn_weights
        
        #7. attention_weights has been generated by the 2 mechanism of attention
        # x has been processed by the 2 blocks (1: dense -> 2:conv)
        
        return x, attention_weights


class CrossAttentionResnorm(tf.keras.layers.Layer):
    
    def __init__(self,
                 model_dim: int,
                 num_heads: int,
                 dropout_rate: float,
                 **kwargs):
        
        super(CrossAttentionResnorm, self).__init__(**kwargs)               # initialize the parent class (=keras.layers.Layer)
        
        self.mha = MultiHeadAttention(model_dim, num_heads)                 # use the multihead attention layer
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)   # use the layer Normalisation
        self.dropout = tf.keras.layers.Dropout(dropout_rate)                # use the DropOut layer
    
    def call(self, q, k, v, training, mask, drop_n_heads):
        
        # 1. contrary to previous the multi head attention is called with Q,K,V
        # this is not self attention!
        attn_values, attn_weights = self.mha(v, k=k, q_in=q, mask=mask, training=training, drop_n_heads=drop_n_heads)
        
        # 2. call the DropOut layer
        attn_values = self.dropout(attn_values, training=training)
        
        #3. call the layer Normalisation
        out = self.layernorm(attn_values + q)
        
        return out, attn_weights


class CrossAttentionDenseBlock(tf.keras.layers.Layer):
    
    def __init__(self,
                 model_dim: int,
                 num_heads: int,
                 dense_hidden_units: int,
                 dropout_rate: float,
                 **kwargs):
        
        super(CrossAttentionDenseBlock, self).__init__(**kwargs)  # initialize the parent class (=keras.layers.Layer)
        
        self.sarn = SelfAttentionResNorm(model_dim, num_heads, dropout_rate=dropout_rate)   # SELF attention + res/skip + norm layer
        self.carn = CrossAttentionResnorm(model_dim, num_heads, dropout_rate=dropout_rate)  # CROSS attention res/skip + norm layer
        self.ffn = FFNResNorm(model_dim, dense_hidden_units, dropout_rate=dropout_rate)     # Dense +  res/skip + Normalisation layer
    
    def call(self, x, enc_output, training, look_ahead_mask, padding_mask, drop_n_heads):
        
        #1. call SELF attention + res/skip + norm layer
        attn1, attn_weights_block1 = self.sarn(x, mask=look_ahead_mask, training=training, drop_n_heads=drop_n_heads)
        
        #2. call CROSS attention res/skip + norm layer
        attn2, attn_weights_block2 = self.carn(attn1, v=enc_output, k=enc_output,
                                               mask=padding_mask, training=training, drop_n_heads=drop_n_heads)
        
        #3. call Dense +  res/skip + Normalisation layer
        ffn_out = self.ffn(attn2, training=training)
        return ffn_out, attn_weights_block1, attn_weights_block2


class CrossAttentionConvBlock(tf.keras.layers.Layer):
    
    def __init__(self,
                 model_dim: int,
                 num_heads: int,
                 conv_filters: int,
                 dropout_rate: float,
                 kernel_size: int,
                 conv_padding: str,
                 conv_activation: str,
                 **kwargs):
        
        super(CrossAttentionConvBlock, self).__init__(**kwargs) # initialize the parent class (=keras.layers.Layer)
        
        self.sarn = SelfAttentionResNorm(model_dim, num_heads, dropout_rate=dropout_rate)
        self.carn = CrossAttentionResnorm(model_dim, num_heads, dropout_rate=dropout_rate)
        self.conv = CNNResNorm(out_size=model_dim,
                               n_layers=2,
                               hidden_size=conv_filters,
                               kernel_size=kernel_size,
                               inner_activation=conv_activation,
                               last_activation=conv_activation,
                               padding=conv_padding,
                               normalization='batch')
    
    def call(self, x, enc_output, training, look_ahead_mask, padding_mask, drop_n_heads):
        attn1, attn_weights_block1 = self.sarn(x, mask=look_ahead_mask, training=training, drop_n_heads=drop_n_heads)
        
        attn2, attn_weights_block2 = self.carn(attn1, v=enc_output, k=enc_output,
                                               mask=padding_mask, training=training, drop_n_heads=drop_n_heads)
        ffn_out = self.conv(attn2, training=training)
        return ffn_out, attn_weights_block1, attn_weights_block2


class CrossAttentionBlocks(tf.keras.layers.Layer):
    
    def __init__(self,
                 model_dim: int,
                 feed_forward_dimension: int,
                 num_heads: list,
                 maximum_position_encoding: int,
                 dropout_rate: float,
                 dense_blocks: int,
                 conv_filters: int,
                 conv_activation: str,
                 conv_padding: str,
                 conv_kernel: int,
                 **kwargs):
        
        super(CrossAttentionBlocks, self).__init__(**kwargs) # initialize the parent class (=keras.layers.Layer)
        
        self.model_dim = model_dim
        
        self.pos_encoding_scalar = tf.Variable(1.)
        self.pos_encoding = positional_encoding(maximum_position_encoding, model_dim)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        
        
        self.CADB = [
            CrossAttentionDenseBlock(model_dim=model_dim, dropout_rate=dropout_rate, num_heads=n_heads,
                                     dense_hidden_units=feed_forward_dimension, name=f'{self.name}_CADB_{i}')
            for i, n_heads in enumerate(num_heads[:dense_blocks])]
        self.CACB = [
            CrossAttentionConvBlock(model_dim=model_dim, dropout_rate=dropout_rate, num_heads=n_heads,
                                    name=f'{self.name}_CACB_{i}', conv_filters=conv_filters,
                                    conv_activation=conv_activation, conv_padding=conv_padding, kernel_size=conv_kernel)
            for i, n_heads in enumerate(num_heads[dense_blocks:])]
    
    
    
    def call(self, inputs, enc_output, training, decoder_padding_mask, encoder_padding_mask, drop_n_heads,
             reduction_factor=1):
        
        seq_len = tf.shape(inputs)[1]
        
        x = inputs * tf.math.sqrt(tf.cast(self.model_dim, tf.float32))
        x += self.pos_encoding_scalar * self.pos_encoding[:, :seq_len * reduction_factor:reduction_factor, :]
        x = self.dropout(x, training=training)
        
        attention_weights = {}
        
        for i, block in enumerate(self.CADB):
            x, _, attn_weights = block(x, enc_output, training, decoder_padding_mask, encoder_padding_mask,
                                       drop_n_heads)
            attention_weights[f'{self.name}_DenseBlock{i + 1}_CrossAttention'] = attn_weights
        
        for i, block in enumerate(self.CACB):
            x, _, attn_weights = block(x, enc_output, training, decoder_padding_mask, encoder_padding_mask,
                                       drop_n_heads)
            attention_weights[f'{self.name}_ConvBlock{i + 1}_CrossAttention'] = attn_weights
        
        return x, attention_weights


class DecoderPrenet(tf.keras.layers.Layer):
    
    def __init__(self,
                 model_dim: int,
                 dense_hidden_units: int,
                 dropout_rate: float,
                 **kwargs):
        super(DecoderPrenet, self).__init__(**kwargs)               # initialize the parent class (=keras.layers.Layer)
        
        
        self.d1 = tf.keras.layers.Dense(dense_hidden_units,
                                        activation='relu')             # (batch_size, seq_len, dense_hidden_units)
        self.d2 = tf.keras.layers.Dense(model_dim, activation='relu')  # (batch_size, seq_len, model_dim)
        
        self.rate = tf.Variable(dropout_rate, trainable=False)
        
        self.dropout_1 = tf.keras.layers.Dropout(self.rate)
        self.dropout_2 = tf.keras.layers.Dropout(self.rate)
    
    def call(self, x):
        
        self.dropout_1.rate = self.rate
        self.dropout_2.rate = self.rate
        
        x = self.d1(x)
        
        # use dropout also in inference for positional encoding relevance
        x = self.dropout_1(x, training=True)
        x = self.d2(x)
        x = self.dropout_2(x, training=True)
        return x


class Postnet(tf.keras.layers.Layer):
    
    def __init__(self, mel_channels: int,
                 conv_filters: int,
                 conv_layers: int,
                 kernel_size: int,
                 **kwargs):
        
        super(Postnet, self).__init__(**kwargs)
        
        self.mel_channels = mel_channels
        
        self.stop_linear = tf.keras.layers.Dense(3)
        
        self.conv_blocks = CNNResNorm(out_size=mel_channels,
                                      kernel_size=kernel_size,
                                      padding='causal',
                                      inner_activation='tanh',
                                      last_activation='linear',
                                      hidden_size=conv_filters,
                                      n_layers=conv_layers,
                                      normalization='batch')
        
        self.add_layer = tf.keras.layers.Add()
    
    def call(self, x, training):
        stop = self.stop_linear(x)
        conv_out = self.conv_blocks(x, training=training)
        return {
            'mel_linear': x,
            'final_output': conv_out,
            'stop_prob': stop,
        }


class DurationPredictor(tf.keras.layers.Layer):
    def __init__(self,
                 model_dim: int,
                 kernel_size: int,
                 conv_padding: str,
                 conv_activation: str,
                 conv_block_n: int,
                 dense_activation: str,
                 **kwargs):
        
        super(DurationPredictor, self).__init__(**kwargs)
        
        self.conv_blocks = CNNResNorm(out_size=model_dim,
                                      kernel_size=kernel_size,
                                      padding=conv_padding,
                                      inner_activation=conv_activation,
                                      last_activation=conv_activation,
                                      hidden_size=model_dim,
                                      n_layers=conv_block_n,
                                      normalization='layer')
        
        self.linear = tf.keras.layers.Dense(1, activation=dense_activation,
                                            bias_initializer=tf.keras.initializers.Constant(value=1))
    
    def call(self, x, training):
        x = self.conv_blocks(x, training=training)
        x = self.linear(x)
        return x


class Expand(tf.keras.layers.Layer):
    """ Expands a 3D tensor on its second axis given a list of dimensions.
        Tensor should be:
            batch_size, seq_len, dimension
        
        E.g:
        input = tf.Tensor([[[0.54710746 0.8943467 ]
                          [0.7140938  0.97968304]
                          [0.5347662  0.15213418]]], shape=(1, 3, 2), dtype=float32)
        dimensions = tf.Tensor([1 3 2], shape=(3,), dtype=int32)
        output = tf.Tensor([[[0.54710746 0.8943467 ]
                           [0.7140938  0.97968304]
                           [0.7140938  0.97968304]
                           [0.7140938  0.97968304]
                           [0.5347662  0.15213418]
                           [0.5347662  0.15213418]]], shape=(1, 6, 2), dtype=float32)
    """
    
    def __init__(self, model_dim, **kwargs):
        super(Expand, self).__init__(**kwargs)
        self.model_dimension = model_dim
    
    def call(self, x, dimensions):
        
        # reduce the dimensions
        dimensions = tf.squeeze(dimensions, axis=-1)
        # round the dimensiosn to the next int
        dimensions = tf.cast(tf.math.round(dimensions), tf.int32)
        # x is of shape (batch_size,seq_len...)
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]
        
        
        # build masks from dimensions
        max_dim = tf.math.reduce_max(dimensions)
        tot_dim = tf.math.reduce_sum(dimensions)
        
        
        index_masks = tf.RaggedTensor.from_row_lengths(tf.ones(tot_dim), tf.reshape(dimensions, [-1])).to_tensor()
        index_masks = tf.cast(tf.reshape(index_masks, (batch_size, seq_len * max_dim)), tf.float32)
        non_zeros = seq_len * max_dim - tf.reduce_sum(max_dim - dimensions, axis=1)
        
        # stack and mask
        tiled = tf.tile(x, [1, 1, max_dim])
        
        reshaped = tf.reshape(tiled, (batch_size, seq_len * max_dim, self.model_dimension))
        
        mask_reshape = tf.multiply(reshaped, index_masks[:, :, tf.newaxis])
        
        ragged = tf.RaggedTensor.from_row_lengths(mask_reshape[index_masks > 0], non_zeros)
        return ragged.to_tensor()
