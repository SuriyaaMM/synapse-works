import { gql } from '@apollo/client/core';

export const GET_MODEL = gql`
  query GetModel($id: ID!) {
    getModel(id: $id) {
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
      }
    }
  }
`;

export const GET_TRAINING_STATUS = gql`
  query GetTrainingStatus {
    getTrainingStatus {
      epoch
      loss
      accuracy
      completed
    }
  }
`;

export const GET_MODELS = gql`
  query GetModels {
    getModels {
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