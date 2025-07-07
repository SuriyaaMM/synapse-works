import { gql } from '@apollo/client/core';

export const GET_MODEL = gql`
  query GetModel {
    getModel {
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

export const VALIDATE_MODEL = gql`
  query ValidateModel($modelId: ID!, $in_dimension: [Int!]!) {
    validateModel(id: $modelId, in_dimension: $in_dimension) {
      status {
        layer_id
        message
        in_dimension
        out_dimension
        required_in_dimension
      }
    }
  }
`;

export const GET_TRAINING_STATUS = gql`
  query GetTrainingStatus {
    getTrainingStatus {
      epoch
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
