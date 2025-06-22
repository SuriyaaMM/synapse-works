import { gql } from '@apollo/client/core';

export const CREATE_MODEL = gql`
  mutation CreateMyTestModel($name: String!) {
    createModel(name: $name) {
      id
      name
      layers_config {
        id
      }
    }
  }
`;

export const APPEND_LINEAR_LAYER = gql`
  mutation AddMyLinearLayer(
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

export const SET_TRAIN_CONFIG = gql`
  mutation SetMyModelTrainConfig(
    $modelId: ID!
    $epochs: Int!
    $optimizer: String!
    $optimizerConfig: OptimizerConfigInput!
    $loss_function: String!
  ) {
    setTrainConfig(
      model_id: $modelId
      train_config: {
        epochs: $epochs
        optimizer: $optimizer
        optimizer_config: $optimizerConfig
        loss_function: $loss_function
      }
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
    $modelId: ID!
    $datasetConfig: DatasetConfigInput!
  ) {
    setDataset(
      model_id: $modelId 
      dataset_config: $datasetConfig
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