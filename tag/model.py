import numpy as np

from keras.optimizers import Adadelta
from keras.models import Graph
from keras.layers.core import Dense, Dropout, Activation, Flatten, Merge
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.convolutional import Convolution1D, MaxPooling1D


def build_model(std_token_len, left_char_len,
                right_char_len, filter_length,
                nb_filters, nb_dense_dims,
                nb_left_tokens, nb_right_tokens,
                nb_labels, char_vector_dict, train_token_index,
                target_representation, context_representation,
                embedding_dims, include_target_one_hot, max_pooling,
                pretrain_embeddings, embeddings):

    m = Graph()

    # specify inputs:
    if target_representation == 'embedding':
        m.add_input(name='target_input', input_shape=(1,), dtype='int')
    else:
        m.add_input(name='target_input', input_shape=(std_token_len, len(char_vector_dict)))

    if context_representation in ('flat_embeddings', 'lstm_embeddings'):
        m.add_input(name='left_input',  input_shape=(1,), dtype='int')
        m.add_input(name='right_input',  input_shape=(1,), dtype='int')
    elif context_representation != 'None':
        m.add_input(name='left_input', input_shape=(left_char_len, len(char_vector_dict)))
        m.add_input(name='right_input', input_shape=(right_char_len, len(char_vector_dict)))

    if include_target_one_hot:
        m.add_input(name='target_one_hot_input', input_shape=(1,), dtype='int')
    
    ### TARGET #################################################################################
    if context_representation == 'flat_characters':
        # add left context nodes:
        m.add_node(Flatten(input_shape=(left_char_len, len(char_vector_dict))),
                           name="left_flatten", input="left_input")
        m.add_node(Dropout(0.5),
                           name="left_dropout1", input="left_flatten")
        m.add_node(Dense(output_dim=nb_dense_dims),
                           name="left_dense", input="left_dropout1")
        m.add_node(Dropout(0.5),
                           name="left_dropout2", input="left_dense")
        m.add_node(Activation('relu'),
                           name='left_out', input='left_dropout2')

        # add right context nodes:
        m.add_node(Flatten(input_shape=(right_char_len, len(char_vector_dict))),
                           name="right_flatten", input="right_input")
        m.add_node(Dropout(0.5),
                           name="right_dropout1", input="right_flatten")
        m.add_node(Dense(output_dim=nb_dense_dims),
                           name="right_dense", input="right_dropout1")
        m.add_node(Dropout(0.5),
                           name="right_dropout2", input="right_dense")
        m.add_node(Activation('relu'),
                           name='right_out', input='right_dropout2')

    elif context_representation == 'lstm_characters':
        # add left context nodes:
        m.add_node(LSTM(input_dim=len(char_vector_dict), output_dim=nb_dense_dims),
                           name='left_lstm1', input='left_input')
        m.add_node(Dropout(0.5),
                           name='left_dropout1', input='left_lstm1')
        m.add_node(Activation('relu'),
                           name='left_out', input='left_dropout1')

        # add right context nodes:
        m.add_node(LSTM(input_dim=len(char_vector_dict), output_dim=nb_dense_dims),
                           name='right_lstm1', input='right_input')
        m.add_node(Dropout(0.5),
                           name='right_dropout1', input='right_lstm1')
        m.add_node(Activation('relu'),
                           name='right_out', input='right_dropout1')

    elif context_representation == 'flat_convolution':
        # add left context nodes:
        m.add_node(Convolution1D(input_dim=len(char_vector_dict),
                                nb_filter=nb_filters,
                                filter_length=filter_length,
                                activation="relu",
                                border_mode="valid",
                                subsample_length=1), 
                       name="left_conv1", input='left_input')
        if max_pooling:
            m.add_node(MaxPooling1D(pool_length=2),
                       name="left_pool1", input="left_conv1")
            m.add_node(Flatten(),
                       name="left_flatten1", input="left_pool1")
            m.add_node(Dropout(0.5),
                       name="left_dropout1", input="left_flatten1")
            m.add_node(Dense(output_dim=nb_dense_dims),
                       name="left_dense1", input="left_dropout1")
            m.add_node(Dropout(0.5),
                       name="left_dropout2", input="left_dense1")
            m.add_node(Activation('relu'),
                       name='left_out', input='left_dropout2')
        else:
            m.add_node(Flatten(),
                       name="left_flatten1", input="left_conv1")
            m.add_node(Dropout(0.5),
                       name="left_dropout1", input="left_flatten1")
            m.add_node(Dense(output_dim=nb_dense_dims),
                       name="left_dense1", input="left_dropout1")
            m.add_node(Dropout(0.5),
                       name="left_dropout2", input="left_dense1")
            m.add_node(Activation('relu'),
                       name='left_out', input='left_dropout2')

        # add right context nodes:
        m.add_node(Convolution1D(input_dim=len(char_vector_dict),
                                nb_filter=nb_filters,
                                filter_length=filter_length,
                                activation="relu",
                                border_mode="valid",
                                subsample_length=1), 
                       name="right_conv1", input='right_input')
        if max_pooling:
            m.add_node(MaxPooling1D(pool_length=2),
                       name="right_pool1", input="right_conv1")
            m.add_node(Flatten(),
                       name="right_flatten1", input="right_pool1")
            m.add_node(Dropout(0.5),
                       name="right_dropout1", input="right_flatten1")
            m.add_node(Dense(output_dim=nb_dense_dims),
                       name="right_dense1", input="right_dropout1")
            m.add_node(Dropout(0.5),
                       name="right_dropout2", input="right_dense1")
            m.add_node(Activation('relu'),
                       name='right_out', input='right_dropout2')
        else:
            m.add_node(Flatten(),
                       name="right_flatten1", input="right_conv1")
            m.add_node(Dropout(0.5),
                       name="right_dropout1", input="right_flatten1")
            m.add_node(Dense(output_dim=nb_dense_dims),
                       name="right_dense1", input="right_dropout1")
            m.add_node(Dropout(0.5),
                       name="right_dropout2", input="right_dense1")
            m.add_node(Activation('relu'),
                       name='right_out', input='right_dropout2')


    elif context_representation == 'lstm_convolution':
        # add left context nodes:
        m.add_node(Convolution1D(input_dim=len(char_vector_dict),
                                nb_filter=nb_filters,
                                filter_length=filter_length,
                                activation="relu",
                                border_mode="valid",
                                subsample_length=1), 
                       name="left_conv1", input='left_input')
        if max_pooling:
            m.add_node(MaxPooling1D(pool_length=2),
                       name="left_pool1", input="left_conv1")
            m.add_node(LSTM(input_dim=nb_filters/2, output_dim=nb_dense_dims),
                       name='left_lstm1', input='left_pool1')
            m.add_node(Dropout(0.5),
                       name='left_dropout1', input='left_lstm1')
            m.add_node(Activation('relu'),
                       name='left_out', input='left_dropout1')
        else:
            m.add_node(LSTM(input_dim=nb_filters, output_dim=nb_dense_dims),
                       name='left_lstm1', input='left_conv1')
            m.add_node(Dropout(0.5),
                       name='left_dropout1', input='left_lstm1')
            m.add_node(Activation('relu'),
                       name='left_out', input='left_dropout1')

        # add right context nodes:
        m.add_node(Convolution1D(input_dim=len(char_vector_dict),
                                nb_filter=nb_filters,
                                filter_length=filter_length,
                                activation="relu",
                                border_mode="valid",
                                subsample_length=1), 
                       name="right_conv1", input='right_input')
        if max_pooling:
            m.add_node(MaxPooling1D(pool_length=2),
                       name="right_pool1", input="right_conv1")
            m.add_node(LSTM(input_dim=nb_filters/2, output_dim=nb_dense_dims),
                       name='right_lstm1', input='right_pool1')
            m.add_node(Dropout(0.5),
                       name='right_dropout1', input='right_lstm1')
            m.add_node(Activation('relu'),
                       name='right_out', input='right_dropout1')
        else:
            m.add_node(LSTM(input_dim=nb_filters, output_dim=nb_dense_dims),
                       name='right_lstm1', input='right_conv1')
            m.add_node(Dropout(0.5),
                       name='right_dropout1', input='right_lstm1')
            m.add_node(Activation('relu'),
                       name='right_out', input='right_dropout1')

    elif context_representation == 'flat_embeddings':
        m.add_node(Embedding(input_dim=len(train_token_index), output_dim=embedding_dims, weights=embeddings, input_length=nb_left_tokens),
                       name='left_embedding', input='left_input')
        m.add_node(Flatten(),
                       name="left_flatten", input="left_embedding")
        m.add_node(Dropout(0.5),
                       name='left_dropout', input='left_flatten')
        m.add_node(Activation('relu'),
                       name='left_relu', input='left_dropout')
        m.add_node(Dense(output_dim=nb_dense_dims),
                       name="left_dense1", input="left_relu")
        m.add_node(Dropout(0.5),
                       name="left_dropout2", input="left_dense1")
        m.add_node(Activation('relu'),
                       name='left_out', input='left_dropout2')

        m.add_node(Embedding(input_dim=len(train_token_index), output_dim=embedding_dims, weights=embeddings, input_length=nb_right_tokens),
                       name='right_embedding', input='right_input')
        m.add_node(Flatten(),
                       name="right_flatten", input="right_embedding")
        m.add_node(Dropout(0.5),
                       name='right_dropout', input='right_flatten')
        m.add_node(Activation('relu'),
                       name='right_relu', input='right_dropout')
        m.add_node(Dense(output_dim=nb_dense_dims),
                       name="right_dense1", input="right_relu")
        m.add_node(Dropout(0.5),
                       name="right_dropout2", input="right_dense1")
        m.add_node(Activation('relu'),
                       name='right_out', input='right_dropout2')

    elif context_representation == 'lstm_embeddings':
        m.add_node(Embedding(input_dim=len(train_token_index), output_dim=embedding_dims, weights=embeddings, input_length=nb_left_tokens),
                       name='left_embedding', input='left_input')
        m.add_node(LSTM(input_dim=embedding_dims, output_dim=nb_dense_dims),
                       name='left_lstm', input='left_embedding')
        m.add_node(Dropout(0.5),
                       name='left_dropout', input='left_lstm')
        m.add_node(Activation('relu'),
                       name='left_out', input='left_dropout')

        m.add_node(Embedding(input_dim=len(train_token_index), output_dim=embedding_dims, weights=embeddings, input_length=nb_right_tokens),
                       name='right_embedding', input='right_input')
        m.add_node(LSTM(input_dim=embedding_dims, output_dim=nb_dense_dims),
                       name='right_lstm', input='right_embedding')
        m.add_node(Dropout(0.5),
                       name='right_dropout', input='right_lstm')
        m.add_node(Activation('relu'),
                       name='right_out', input='right_dropout')
    
    ### TARGET #################################################################################
    if target_representation == 'flat_characters':
        m.add_node(Flatten(input_shape=(std_token_len, len(char_vector_dict))),
                   name="token_flatten", input="target_input")
        m.add_node(Dropout(0.5),
                   name="token_dropout1", input="token_flatten")
        m.add_node(Dense(output_dim=nb_dense_dims),
                   name="target_dense", input="token_dropout1")
        m.add_node(Dropout(0.5),
                   name="target_dropout2", input="target_dense")
        m.add_node(Activation('relu'),
                   name='target_out', input='target_dropout2')

    elif target_representation == 'lstm_characters':
        m.add_node(LSTM(input_dim=len(char_vector_dict), output_dim=nb_dense_dims),
                   name='target_lstm', input='target_input')
        m.add_node(Dropout(0.5),
                   name='target_dropout', input='target_lstm')
        m.add_node(Activation('relu'),
                   name='target_out', input='target_dropout')

    elif target_representation == 'flat_convolution':
        # add target token nodes:
        m.add_node(Convolution1D(input_dim=len(char_vector_dict),
                                nb_filter=nb_filters,
                                filter_length=filter_length,
                                activation="relu",
                                border_mode="valid",
                                subsample_length=1), 
                       name="target_conv1", input='target_input')
        if max_pooling:
            m.add_node(MaxPooling1D(pool_length=2),
                       name="target_pool1", input="target_conv1")
            m.add_node(Flatten(),
                       name="target_flatten1", input="target_pool1")
            m.add_node(Dropout(0.5),
                       name="target_dropout1", input="target_flatten1")
            m.add_node(Dense(output_dim=nb_dense_dims),
                       name="target_dense1", input="target_dropout1")
            m.add_node(Dropout(0.5),
                       name="target_dropout2", input="target_dense1")
            m.add_node(Activation('relu'),
                       name='target_out', input='target_dropout2')
        else:
            m.add_node(Flatten(),
                       name="target_flatten1", input="target_conv1")
            m.add_node(Dropout(0.5),
                       name="target_dropout1", input="target_flatten1")
            m.add_node(Dense(output_dim=nb_dense_dims),
                       name="target_dense1", input="target_dropout1")
            m.add_node(Dropout(0.5),
                       name="target_dropout2", input="target_dense1")
            m.add_node(Activation('relu'),
                       name='target_out', input='target_dropout2')

    elif target_representation == 'lstm_convolution':
        m.add_node(Convolution1D(input_dim=len(char_vector_dict),
                                nb_filter=nb_filters,
                                filter_length=filter_length,
                                activation="relu",
                                border_mode="valid",
                                subsample_length=1), 
                       name="target_conv1", input='target_input')
        if max_pooling:
            m.add_node(MaxPooling1D(pool_length=2),
                       name="target_pool1", input="target_conv1")
            m.add_node(LSTM(input_dim=nb_filters/2,
                       output_dim=nb_dense_dims),
                       name='target_lstm1', input='target_pool1')
            m.add_node(Dropout(0.5),
                       name='target_dropout1', input='target_lstm1')
            m.add_node(Activation('relu'),
                       name='target_out', input='target_dropout1')
        else:
            m.add_node(LSTM(input_dim=nb_filters, output_dim=nb_dense_dims),
                       name='target_lstm1', input='target_conv1')
            m.add_node(Dropout(0.5),
                       name='target_dropout1', input='target_lstm1')
            m.add_node(Activation('relu'),
                       name='target_out', input='target_dropout1')

    elif target_representation == 'embedding':
        m.add_node(Embedding(input_dim=len(train_token_index), output_dim=nb_dense_dims, input_length=1),
                   name='target_embedding', input='target_input')
        m.add_node(Flatten(),
                   name="target_flatten", input="target_embedding")
        m.add_node(Dropout(0.5),
                   name='target_dropout', input='target_flatten')
        m.add_node(Activation('relu'),
                   name='target_out', input='target_dropout')


    if context_representation == "None":
        if include_target_one_hot:
            m.add_node(Embedding(input_dim=len(train_token_index), output_dim=nb_dense_dims, input_length=1),
                                 name='target_one_hot_embedding', input='target_one_hot_input')
            m.add_node(Flatten(),
                                 name="target_one_hot_flatten", input="target_one_hot_embedding")
            m.add_node(Dropout(0.5),
                                 name='target_one_hot_dropout', input='target_one_hot_flatten')
            m.add_node(Activation('relu'),
                                 name='target_one_hot_out', input='target_one_hot_dropout')

            m.add_node(Dense(output_dim=nb_labels),
                       name='label_dense',
                       inputs=['target_out', 'target_one_hot_dropout'],
                       merge_mode='concat')
        else:
            m.add_node(Dense(output_dim=nb_labels),
                       name='label_dense',
                       input='target_out')
    else:
        if include_target_one_hot:
            m.add_node(Embedding(input_dim=len(train_token_index), output_dim=nb_dense_dims, input_length=1),
                        name='target_one_hot_embedding', input='target_one_hot_input')
            m.add_node(Flatten(),
                        name="target_one_hot_flatten", input="target_one_hot_embedding")
            m.add_node(Dropout(0.5),
                        name='target_one_hot_dropout', input='target_one_hot_flatten')
            m.add_node(Activation('relu'),
                        name='target_one_hot_out', input='target_one_hot_dropout')

            m.add_node(Dense(output_dim=nb_labels),
                       name='label_dense',
                       inputs=['target_out', 'left_out', 'right_out', 'target_one_hot_out'],
                       merge_mode='concat')
        else:
            m.add_node(Dense(output_dim=nb_labels),
                       name='label_dense',
                       inputs=['target_out', 'left_out', 'right_out'],
                       merge_mode='concat')


    m.add_node(Dropout(0.5),
                   name="label_dropout", input="label_dense")
    m.add_node(Activation('softmax'),
                   name='label_softmax', input='label_dropout')

    m.add_output(name='label_output', input='label_softmax')

    m.compile(optimizer='adadelta',
              loss={'label_output':'categorical_crossentropy'})

    return m