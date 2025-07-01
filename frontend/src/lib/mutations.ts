import { gql } from '@apollo/client/core';

export const CREATE_MODEL = gql`
  mutation CreateMyTestModel($name: String!) {
    createModel(name: $name) {
      id
      name
    }
  }
`;

export const APPEND_LAYER = gql`
  mutation AddLayer(
    $modelId: ID!
    $layerConfig: LayerConfigInput!
  ) {
    appendLayer(
      model_id: $modelId
      layer_config: $layerConfig
    ) {
      id
      name
      layers_config {
        id
        type
        name
        ... on LinearLayerConfig {
          in_features
          out_features
        }
      }
    }
  }
`;

export const DELETE_LAYER = gql`
  mutation DeleteLayer($model_id: ID!, $layer_id: ID!) {
    deleteLayer(model_id: $model_id, layer_id: $layer_id) {
      id
      name
      layers_config {
        id
        type
        ... on LinearLayerConfig {
          name
          in_features
          out_features
        }
      }
    }
  }
`;

export const MODIFY_LAYER = gql`
  mutation ModifyLayer($model_id: ID!, $layer_id: ID!, $layer_config: LayerConfigInput!) {
    modifyLayer(model_id: $model_id, layer_id: $layer_id, layer_config: $layer_config) {
      id
      name
      layers_config {
        id
        type
        ... on LinearLayerConfig {
          name
          in_features
          out_features
        }
      }
    }
  }
`;

export const SET_TRAIN_CONFIG = gql`
  mutation SetMyModelTrainConfig(
    $epochs: Int!
    $optimizer: String!
    $optimizerConfig: OptimizerConfigInput!
    $loss_function: String!
  ) {
    setTrainConfig(
      train_config: {
        epochs: $epochs
        optimizer: $optimizer
        optimizer_config: $optimizerConfig
        loss_function: $loss_function
      }
    ) {
      id
      name
      train_config { 
        epochs
        optimizer
        optimizer_config {
          lr
        }
        loss_function
      }
    }
  }
`;

export const SET_DATASET_CONFIG = gql`
  mutation SetMyDatasetConfig(
    $datasetConfig: DatasetConfigInput!
  ) {
    setDataset(
      dataset_config: $datasetConfig
    ) {
      id
      name
      train_config { 
        epochs
        optimizer
        optimizer_config {
          lr
        }
        loss_function
      }
      dataset_config {
        name
        batch_size
        split_length
        shuffle
        ... on MNISTDatasetConfig {
          root
          train
          download
        }
      }
    }
  }
`;

export const TRAIN_MODEL = gql`
  mutation TrainMyModel($modelId: ID!) {
    train(model_id: $modelId) {
      id
      name
      layers_config {
        id
        type
        name
        ... on LinearLayerConfig { 
          in_features
          out_features
        }
      }
      train_config {
        epochs
        optimizer
        optimizer_config {
          lr
        }
        loss_function
      }
      dataset_config {
        name
        batch_size
        split_length
        shuffle
        ... on MNISTDatasetConfig {
          root
          train
          download
        }
      }
    }
  }
`;

export const START_TENSORBOARD = gql`
  mutation StartTensorboard {
    startTensorboard
  }
`;

export const SAVE_MODEL = gql`
  mutation SaveModel {
    saveModel
  }
`;

export const LOAD_MODEL = gql`
  mutation LoadModel($modelId: ID!){
    loadModel(model_id: $modelId) {
      id
      name
      train_config { 
        epochs
        optimizer
        optimizer_config {
          lr
        }
        loss_function
      }
      dataset_config {
        name
        batch_size
        split_length
        shuffle
      }
    }
  }
`;

export const ADD_TO_GRAPH = gql`
  mutation AppendLayer($layer_config: LayerConfigInput!) {
    appendToModuleGraph(layer_config: $layer_config) {
      layers {
        id
        name
        type
        ... on Conv2dLayerConfig { in_channels, out_channels, kernel_size, padding }
        ... on Conv1dLayerConfig { in_channels, out_channels, kernel_size, padding }
        ... on LinearLayerConfig { in_features, out_features }
      }
      edges { source_id, target_ids }
    }
  }
`;

export const CONNECT_NODES = gql`
  mutation ConnectLayers($source_layer_id: ID!, $target_layer_id: ID!) {
    connectInModuleGraph(source_layer_id: $source_layer_id, target_layer_id: $target_layer_id) {
      layers { id, name, type }
      edges { source_id, target_ids }
      sorted
    }
  }
`;

export const DISCONNECT_NODES = gql`
  mutation DisconnectLayers($source_layer_id: ID!, $target_layer_id: ID!) {
    disconnectInModuleGraph(source_layer_id: $source_layer_id, target_layer_id: $target_layer_id) {
      layers { id, name, type }
      edges { source_id, target_ids }
      sorted
    }
  }
`;

export const DELETE_FROM_GRAPH = gql`
  mutation DeleteLayer($layer_id: ID!) {
    deleteInModuleGraph(layer_id: $layer_id) {
      layers { id, name, type }
      edges { source_id, target_ids }
      sorted
    }
  }
`;

export const BUILD_MODULE_GRAPH = gql`
  mutation ConstructGraph {
    buildModuleGraph {
      id
      name
      module_graph {
        layers { 
          id
          type
          name
        }
        edges { 
          source_id
          target_ids
        }
        sorted
      }
    }
  }
`;

export const VALIDATE_GRAPH = gql`
  mutation ValidateGraph($in_dimension: [Int!]!) {
  validateModuleGraph(in_dimension: $in_dimension) {
    status {
      message
      out_dimension
      required_in_dimension
    }
  }
}
`;