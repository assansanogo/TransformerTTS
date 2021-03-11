import tensorflow as tf


def new_scaled_crossentropy(index=2, scaling=1.0):
    """
    Returns masked crossentropy with extra scaling:
    Scales the loss for given stop_index by stop_scaling
    """
    
    def masked_crossentropy(targets: tf.Tensor, logits: tf.Tensor) -> tf.Tensor:
        
        crossentropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        #1 when target =0
        padding_mask = tf.math.equal(targets, 0)
        # 0 when target =0
        padding_mask = tf.math.logical_not(padding_mask)
        # convert to float 32
        padding_mask = tf.cast(padding_mask, dtype=tf.float32)
        # equal to  1 @ stop index
        stop_mask = tf.math.equal(targets, index)
        # correction in case of scaling factor
        stop_mask = tf.cast(stop_mask, dtype=tf.float32) * (scaling - 1.)
        
        combined_mask = padding_mask + stop_mask
        loss = crossentropy(targets, logits, sample_weight=combined_mask)
        return loss
    
    return masked_crossentropy


def masked_crossentropy(targets: tf.Tensor, logits: tf.Tensor) -> tf.Tensor:
    
    # normal scc
    crossentropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    # turn the mask into 1111 000 (1 where target and mask_value are different)
    mask = tf.math.logical_not(tf.math.equal(targets, 0))
    # transform the mask  format (=cast) into integers
    mask = tf.cast(mask, dtype=tf.int32)
    #allows to have a weight of zero if target = 0000
    loss = crossentropy(targets, logits, sample_weight=mask)
    return loss


def masked_mean_squared_error(targets: tf.Tensor, logits: tf.Tensor) -> tf.Tensor:
    # misleading name
    # normal mse
    mse = tf.keras.losses.MeanSquaredError()
    # turn the mask into 1111 000 (1 where target and mask_value are different)
    mask = tf.math.logical_not(tf.math.equal(targets, 0))
    # transform the mask  format (=cast) into integers
    mask = tf.cast(mask, dtype=tf.int32)
    # find the maximum value of the mask according to the last dimension
    mask = tf.reduce_max(mask, axis=-1)
    # weight samples allows a weight of zero if target = 0000
    loss = mse(targets, logits, sample_weight=mask)
    return loss


def masked_mean_absolute_error(targets: tf.Tensor, logits: tf.Tensor, mask_value=0,
                               mask: tf.Tensor = None) -> tf.Tensor:
    # misleading name
    # normal mae
    mae = tf.keras.losses.MeanAbsoluteError()
    # corrected by a mask
    if mask is not None:
        # turn the mask into 1111 000 (1 where target and mask_value are different)
        mask = tf.math.logical_not(tf.math.equal(targets, 
                                                 mask_value))
        # transform the mask  format (=cast) into integers
        mask = tf.cast(mask, dtype=tf.int32)
        # find the maximum value of the mask
        mask = tf.reduce_max(mask, axis=-1)
    #allows to have a weight of zero if target = 0000
    loss = mae(targets, logits, sample_weight=mask)
    return loss


def masked_binary_crossentropy(targets: tf.Tensor, logits: tf.Tensor, mask_value=-1) -> tf.Tensor:
    #normal bc
    bc = tf.keras.losses.BinaryCrossentropy(reduction='none')
    # turn the mask into 1111 000 (1 where target and mask_value are different)
    mask = tf.math.logical_not(tf.math.equal(logits,
                                             mask_value))  
    # TODO: masking based on the logits requires a masking layer. But masking layer produces 0. as outputs.
    # Need explicit masking
    
    # transform the mask  format (=cast) into integers
    mask = tf.cast(mask, dtype=tf.int32)
    # bc loss weighted by mask 
    loss_ = bc(targets, logits)
    loss_ *= mask
    return tf.reduce_mean(loss_)


def weighted_sum_losses(targets, pred, loss_functions, coeffs):
    total_loss = 0
    loss_vals = []
    # loop over values of different loss (weighted sum)
    for i in range(len(loss_functions)):
        loss = loss_functions[i](targets[i], pred[i])
        loss_vals.append(loss)
        #weighted sum
        total_loss += coeffs[i] * loss
    return total_loss, loss_vals
