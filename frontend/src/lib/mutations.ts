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
    $loss_function_config: LossFunctionConfigInput
    $metrics: TrainMetricsInput!
  ) {
    setTrainConfig(
      train_config: {
        epochs: $epochs
        optimizer: $optimizer
        optimizer_config: $optimizerConfig
        loss_function: $loss_function
        loss_function_config: $loss_function_config
        metrics: $metrics
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
        loss_function_config {
          reduction
          ignore_index
          label_smoothing
        }
        metrics {
          gradient_visualization
          gradient_norm_visualization
          learning_rate_visualization
          weights_visualization
          graph_visualization
          profile
          accuracy_visualization
          loss_visualization
          test_validation
          test_validation_period
          train_validation
          train_validation_period
        }
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
  mutation TrainModel($args: TrainArgs!) {
    train(args: $args) {
      id
      name
      train_config { epochs }
      dataset_config { name }
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
        ... on LinearLayerConfig { in_features out_features }

        ... on Conv2dLayerConfig { in_channels out_channels kernel_size }
        ... on Conv1dLayerConfig { in_channels out_channels kernel_size }
        ... on ConvTranspose2dLayerConfig { in_channels out_channels kernel_size }

        ... on MaxPool2dLayerConfig { kernel_size }
        ... on MaxPool1dLayerConfig { kernel_size }

        ... on AvgPool2dLayerConfig { kernel_size }
        ... on AvgPool1dLayerConfig { kernel_size }

        ... on BatchNorm2dLayerConfig { num_features }
        ... on BatchNorm1dLayerConfig { num_features }

        ... on DropoutLayerConfig { p }
        ... on Dropout2dLayerConfig { p }

        ... on ELULayerConfig { alpha }
        ... on LeakyReLULayerConfig { negative_slope }

        ... on CatLayerConfig { dimension }

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