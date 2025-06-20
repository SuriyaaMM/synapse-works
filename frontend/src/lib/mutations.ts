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
    $type: String!
    $inFeatures: Int!
    $outFeatures: Int!
    $name: String!
  ) {
    appendLayer(
      model_id: $modelId
      layer_config: {
        type: $type
        linear: {
          in_features: $inFeatures
          out_features: $outFeatures
          name: $name
        }
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
    }
  }
`;

export const SET_TRAIN_CONFIG = gql`
  mutation SetMyModelTrainConfig(
    $modelId: ID!
    $epochs: Int!
    $optimizer: String!
    $lr: Float!
    $loss_function: String!
  ) {
    setTrainConfig(
      model_id: $modelId
      train_config: {
        epochs: $epochs
        optimizer: $optimizer
        optimizer_config: {
          lr: $lr
        }
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
    $name: String!
    $batchSize: Int!
    $shuffle: Boolean!
    $splitLength: [Float!]!
    $mnistRoot: String!
    $mnistTrain: Boolean!
    $mnistDownload: Boolean!
  ) {
    setDataset(
      model_id: $modelId 
      dataset_config: {
        name: $name
        batch_size: $batchSize
        shuffle: $shuffle
        split_length: $splitLength            
        mnist: {
          root: $mnistRoot
          train: $mnistTrain
          download: $mnistDownload
        }
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