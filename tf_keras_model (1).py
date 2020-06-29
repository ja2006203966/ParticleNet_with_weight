import tensorflow as tf
from tensorflow import keras


# A shape is (N, P_A, C), B shape is (N, P_B, C)
# D shape is (N, P_A, P_B)
def batch_distance_matrix_general(A, B):
    with tf.name_scope('dmat'):
        r_A = tf.reduce_sum(A * A, axis=2, keepdims=True)
        r_B = tf.reduce_sum(B * B, axis=2, keepdims=True)
        m = tf.matmul(A, tf.transpose(B, perm=(0, 2, 1)))
        D = r_A - 2 * m + tf.transpose(r_B, perm=(0, 2, 1))
        return D
    
     
    

def knn(num_points, k, topk_indices, features):
    # topk_indices: (N, P, K)
    # features: (N, P, C)
    with tf.name_scope('knn'):
        queries_shape = tf.shape(features)
        batch_size = queries_shape[0]
        batch_indices = tf.tile(tf.reshape(tf.range(batch_size), (-1, 1, 1, 1)), (1, num_points, k, 1))
        indices = tf.concat([batch_indices, tf.expand_dims(topk_indices, axis=3)], axis=3)  # (N, P, K, 2)
        return tf.gather_nd(features, indices)

##====================================================================================================

## write the gpu function
def myweight(A,B,w=-1, actype = 'max'):
    with tf.name_scope('myweight'):
        w = w*2
        w = tf.cast(w, dtype = tf.float32)
        b = tf.tile(A,[1,1,100])
        b = tf.expand_dims(b,-1)
        c = tf.transpose(B, perm= (0, 2, 1))
        c = tf.tile(c,[1,100,1])
        c = tf.expand_dims(c,-1)
        W = tf.concat([b,c] ,-1)
        if actype == 'max':
            W = tf.reduce_max(W, axis=-1)
        if actype == 'min':
            W = tf.reduce_min(W, axis=-1)
        if actype == 'mean':
            W = tf.reduce_mean(W, axis=-1)
#         W = tf.reduce_min(W, axis=-1)
        if actype == 'none':
            W = 1
        if w < 0 :
            W = 1/W
            w = -w
        
        return tf.math.pow(W,w)
        
#         if points.shape[0] is not None:
#             testc = tf.zeros((points.get_shape[0].shape[0],1,100)) +1
#             testb = tf.transpose(testc, perm=(0, 2, 1))
#             c = tf.reshape(c*testc, (points.get_shape[0],100,100,1))
#             b = tf.reshape(b*testb, (points.get_shape[0],100,100,1))
#             W = tf.concat([c, b],  tf.constant(c.get_shape()).get_shape()[-1] -1 )
#             W = tf.math.reduce_max(W, axis= tf.constant(W.get_shape()).get_shape()[-1] -1)
#             W = W 
        
        
    
        
    

# @tf.function
# def mydeltaphi(p1,p2):
#     phi = p1-p2
#     if phi<3.1415926535:
#         phi = phi+2*3.1415926535
#     if phi>=3.1415926535:
#         phi = phi-2*3.1415926535
#     return phi
# @tf.function
# def dij(vec1, vec2, R=0.4, jetpt=-1,w=1, d=1, actype='min'):
#     distance = ((vec1[0]-vec2[0])**2+mydeltaphi(vec1[1],vec2[1])**2)**0.5/R
#     if (d!=0):
#         distance = distance**d
#     if (d==0):
#         distance = 1
#     if (d<0):
#         print('error d must >=0')
#     if (jetpt!=-1):
#         if (actype=='max'):
#             weight = tf.math.maximum(vec1[2],vec2[2])/jetpt
#         if (actype=='min'):
#             weight = tf.math.minimum(vec1[2],vec2[2])/jetpt
#         if (actype=='avg'):
#             weight = (vec1[2]+vec2[2])/(2*jetpt)
#         distance = distance*weight**w    
#     return distance
# @tf.function
# def neighborhood(vec,k=-1,R=0.4,jetpt=-1,w=1,d=1, actype='max'):
#     points = vec.get_shape().as_list()[1]
# #     if k == -1:
# #         k = points
# #     if k > points:
# #         print("You need check k < number of particles")
# #     else:
#     dijset = [[dij(vec[i], vec[j], R, jetpt, w, d, actype=actype) for j in range(points)] for i in range(points)] 
#     x = []
#     index0 = [i for i in range(points)]
#     for i in range(points):
#         edges = tf.sort(dijset[i])
#         index = []
#         for j in edges[:k+1]:
#             index = tf.concat([index,tf.gather(index0,tf.where(tf.equal(dijset[i], j))[0])], 0)                
#         x.append(index)
        

#     return  x 
##=======================================================================================================
def edge_conv(points, ptw, features, num_points, K, channels, w = -1, with_bn=True, activation='relu', pooling='average', name='edgeconv', actype = 'max'):
    """EdgeConv
    Args:
        K: int, number of neighbors
        in_channels: # of input channels
        channels: tuple of output channels
        pooling: pooling method ('max' or 'average')
    Inputs:
        points: (N, P, C_p)
        features: (N, P, C_0)
    Returns:
        transformed points: (N, P, C_out), C_out = channels[-1]
    """ 

    with tf.name_scope('edgeconv'):
     
        # distance
        D = batch_distance_matrix_general(points, points)  # (N, P, P)
#         if points.shape[0] is not None:
        W = myweight(ptw,ptw,w= w, actype = actype)
        D = D*W
        _, indices = tf.nn.top_k(-D, k=K + 1)  # (N, P, K+1)
        indices = indices[:, :, 1:]  # (N, P, K)
#         indices = tf.constant(neighborhood(points, k=K, jetpt = 1 , w=-1, actype='min')  )
        fts = features
        knn_fts = knn(num_points, K, indices, fts)  # (N, P, K, C)
        knn_fts_center = tf.tile(tf.expand_dims(fts, axis=2), (1, 1, K, 1))  # (N, P, K, C)
#         knn_fts_exp2 = tf.math.exp(tf.subtract(knn_fts, knn_fts_center)**2)
        knn_fts = tf.concat([knn_fts_center, tf.subtract(knn_fts, knn_fts_center)], axis=-1)  # (N, P, K, 2*C)

#         knn_fts = tf.concat([knn_fts_center, knn_fts_exp2], axis=-1)  # (N, P, K, 2*C)

        x = knn_fts
        
        for idx, channel in enumerate(channels):
            x = keras.layers.Conv2D(channel, kernel_size=(1, 1), strides=1, data_format='channels_last',
                                    use_bias=False if with_bn else True, kernel_initializer='glorot_normal', name='%s_conv%d' % (name, idx))(x)
            
            if with_bn:
                x = keras.layers.BatchNormalization(name='%s_bn%d' % (name, idx))(x)
            if activation:
                x = keras.layers.Activation(activation, name='%s_act%d' % (name, idx))(x)

        if pooling == 'max':
            fts = tf.reduce_max(x, axis=2)  # (N, P, C')
        else:
            fts = tf.reduce_mean(x, axis=2)  # (N, P, C')

        # shortcut
        sc = keras.layers.Conv2D(channels[-1], kernel_size=(1, 1), strides=1, data_format='channels_last',
                                 use_bias=False if with_bn else True, kernel_initializer='glorot_normal', name='%s_sc_conv' % name)(tf.expand_dims(features, axis=2))
        if with_bn:
            sc = keras.layers.BatchNormalization(name='%s_sc_bn' % name)(sc)
        sc = tf.squeeze(sc, axis=2)

        if activation:
            return keras.layers.Activation(activation, name='%s_sc_act' % name)(sc + fts)  # (N, P, C')
        else:
            return sc + fts


def _particle_net_base(points, features=None, mask=None, setting=None, name='particle_net', w=-1, actype = 'max'):
    # points : (N, P, C_coord)
    # features:  (N, P, C_features), optional
    # mask: (N, P, 1), optinal

    
    with tf.name_scope(name):
        if features is None:
            features = points

        if mask is not None:
            mask = tf.cast(tf.not_equal(mask, 0), dtype='float32')  # 1 if valid
            coord_shift = tf.multiply(999., tf.cast(tf.equal(mask, 0), dtype='float32'))  # make non-valid positions to 99

        fts = tf.squeeze(keras.layers.BatchNormalization(name='%s_fts_bn' % name)(tf.expand_dims(features, axis=2)), axis=2)
#         w =  tf.Variable(-1, trainable = True, name = 'w')
#         w = -1
#         pts0 = tf.squeeze(keras.layers.BatchNormalization(name='%s_pts_bn' % name)(tf.expand_dims(points[:, :, :2], axis=2)), axis=2)
        for layer_idx, layer_param in enumerate(setting.conv_params):
            K, channels = layer_param
#             pts = tf.add(coord_shift, points[:, :, :2]) if layer_idx == 0 else tf.add(coord_shift, fts)
            
            pts = tf.add(coord_shift, points) if layer_idx == 0 else tf.add(coord_shift, fts)

#             pts = tf.add(coord_shift, points[:, :, :2]) if layer_idx == 0 else tf.add(coord_shift, pts0)
#             pts0 = edge_conv(pts, points[:, :, 2:], pts, setting.num_points, K, channels, with_bn=True, activation='relu',
#                             pooling=setting.conv_pooling, name='%s_%s%d' % (name, 'EdgeConv2', layer_idx)) 
#             w0 = tf.Variable(w, trainable = True, dtype=tf.float32, name = 'w0')
            fts = edge_conv(pts, tf.expand_dims(fts[:, :, -1], axis = -1), fts, setting.num_points, K, channels, w = w, with_bn=True, activation='relu', pooling=setting.conv_pooling, name='%s_%s%d' % (name, 'EdgeConv', layer_idx), actype = actype) 

        if mask is not None:
            fts = tf.multiply(fts, mask)

        pool = tf.reduce_mean(fts, axis=1)  # (N, C)
        
         
        if setting.fc_params is not None:
            x = pool
            for layer_idx, layer_param in enumerate(setting.fc_params):
                units, drop_rate = layer_param
#                 w1 = tf.tile([w0],[units])
#                 w1 = tf.expand_dims(w1, axis = 0)
#                 w1 = tf.tile(w1,[tf.shape(x)[0],1])
#                 x = tf.concat([x,w1], axis=-1)
                x = keras.layers.Dense(units, activation='relu')(x)
                if drop_rate is not None and drop_rate > 0:
                    x = keras.layers.Dropout(drop_rate)(x)
            out = keras.layers.Dense(setting.num_class, activation='softmax')(x)
            return out  # (N, num_classes)
        else:
            return pool


class _DotDict:
    pass


def get_particle_net(num_classes, input_shapes , w = -1, actype = 'max'):
    r"""ParticleNet model from `"ParticleNet: Jet Tagging via Particle Clouds"
    <https://arxiv.org/abs/1902.08570>`_ paper.
    Parameters
    ----------
    num_classes : int
        Number of output classes.
    input_shapes : dict
        The shapes of each input (`points`, `features`, `mask`).
    """
    setting = _DotDict()
    setting.num_class = num_classes
    # conv_params: list of tuple in the format (K, (C1, C2, C3))
    setting.conv_params = [
        (16, (64, 64, 64)),
        (16, (128, 128, 128)),
        (16, (256, 256, 256))
        
#         (3, (64, 64, 64)),
#         (3, (64, 64, 64)),
#         (3, (128, 128, 128)),
#         (3, (128, 128, 128)),
#         (3, (256, 256, 256))
        
        ]
    # conv_pooling: 'average' or 'max'
    setting.conv_pooling = 'average'
    # fc_params: list of tuples in the format (C, drop_rate)
    setting.fc_params = [(256, 0.1)]
    setting.num_points = input_shapes['points'][0]

    points = keras.Input(name='points', shape=input_shapes['points'])
    features = keras.Input(name='features', shape=input_shapes['features']) if 'features' in input_shapes else None
    mask = keras.Input(name='mask', shape=input_shapes['mask']) if 'mask' in input_shapes else None
    outputs = _particle_net_base(points, features, mask, setting, name='ParticleNet', w=w, actype = actype)

    return keras.Model(inputs=[points, features, mask], outputs=outputs, name='ParticleNet')


def get_particle_net_lite(num_classes, input_shapes):
    r"""ParticleNet-Lite model from `"ParticleNet: Jet Tagging via Particle Clouds"
    <https://arxiv.org/abs/1902.08570>`_ paper.
    Parameters
    ----------
    num_classes : int
        Number of output classes.
    input_shapes : dict
        The shapes of each input (`points`, `features`, `mask`).
    """
    setting = _DotDict()
    setting.num_class = num_classes
    # conv_params: list of tuple in the format (K, (C1, C2, C3))
    setting.conv_params = [
        (7, (32, 32, 32)),
        (7, (64, 64, 64)),
        ]
    # conv_pooling: 'average' or 'max'
    setting.conv_pooling = 'average'
    # fc_params: list of tuples in the format (C, drop_rate)
    setting.fc_params = [(128, 0.1)]
    setting.num_points = input_shapes['points'][0]

    points = keras.Input(name='points', shape=input_shapes['points'])
    features = keras.Input(name='features', shape=input_shapes['features']) if 'features' in input_shapes else None
    mask = keras.Input(name='mask', shape=input_shapes['mask']) if 'mask' in input_shapes else None
    outputs = _particle_net_base(points, features, mask, setting, name='ParticleNet')

    return keras.Model(inputs=[points, features, mask], outputs=outputs, name='ParticleNet')
