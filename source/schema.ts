export const typeDefs = `#graphql
    # ---------- Layer Config ----------
    interface LayerConfig {
        id: ID! 
        type: String! 
        name: String 
    }
    type LinearLayerConfig implements LayerConfig {
        id: ID! 
        type: String!   
        name: String    
        in_features: Int!  
        out_features: Int! 
    }
    type Conv2dLayerConfig implements LayerConfig {
        id: ID!
        type: String!
        name: String
        in_channels: Int!
        out_channels: Int!
        kernel_size: [Int!]!
        stride: [Int]
        padding: [Int]
        dilation: [Int]
        groups: [Int]
        bias: Boolean
        padding_mode: String
    }
    
    # ---------- Layer Config Input ----------
    input LinearLayerConfigInput {
        name: String
        in_features: Int!      
        out_features: Int!
        bias: Boolean
    }
    input Conv2dLayerConfigInput {
        name: String
        in_channels: Int!
        out_channels: Int!
        kernel_size: [Int!]!
        stride: [Int]
        padding: [Int]
        dilation: [Int]
        groups: [Int]
        bias: Boolean
        padding_mode: String
    }
    # LayerConfig input for collective layers
    input LayerConfigInput {
        type: String! 
        linear: LinearLayerConfigInput
        conv2d: Conv2dLayerConfigInput
    }
    # ---------- Train Config ----------
    type OptimizerConfig {
        lr: Float!
    }
    type TrainConfig {
        epochs: Int!
        optimizer: String!
        optimizer_config: OptimizerConfig!
        loss_function: String!
    }
    type TrainStatus {
        epoch: Int!
        loss: Float!
        accuracy: Float!
        completed: Boolean!
    }
    input OptimizerConfigInput {
        lr: Float!
    }
    input TrainConfigInput {
        epochs: Int!
        optimizer: String!
        optimizer_config: OptimizerConfigInput!
        loss_function: String!
    }
    # ---------- Dataset Config ----------
    interface DatasetConfig {
        name: String!
        batch_size: Int
        split_length: [Float]
        shuffle: Boolean
    }
    type MNISTDatasetConfig implements DatasetConfig {
        name: String!
        batch_size: Int
        split_length: [Float]
        transform: [String]
        shuffle: Boolean
        root: String!
        train: Boolean
        download: Boolean
    }
    type CIFAR10DatasetConfig implements DatasetConfig {
        name: String!
        batch_size: Int
        split_length: [Float]
        shuffle: Boolean
        transform: [String]
        root: String!
        train: Boolean
        download: Boolean
    }
    input MNISTDatasetConfigInput {
        root: String!
        train: Boolean
        download: Boolean
        transform: [String]
    }
    input CIFAR10DatasetConfigInput {
        root: String!
        train: Boolean
        download: Boolean
        transform: [String]
    }
    input DatasetConfigInput {
        name: String!
        batch_size: Int
        split_length: [Float!]
        shuffle: Boolean
        mnist: MNISTDatasetConfigInput
        cifar10: CIFAR10DatasetConfigInput
    }
    # ---------- Model ----------
    # Model type
    type Model {
        id: ID!
        name: String!       
        layers_config: [LayerConfig!]! 
        train_config: TrainConfig!
        dataset_config: DatasetConfig!
    }
    # ---------- Queries ----------
    type Query {
        # get the model by id
        getModel(id: ID!): Model!
        # get all models
        getModels: [Model!]!
        # get training status
        getTrainingStatus: TrainStatus!
    }
    # ---------- Mutations ----------
    type Mutation {
        # create model
        createModel(name: String!): Model!
        # append layer to model
        appendLayer(
            model_id: ID!
            layer_config: LayerConfigInput!
        ): Model!
        # set's training configuration
        setTrainConfig(
            model_id: ID!
            train_config: TrainConfigInput!
        ): Model!
        # set's dataset configuration
        setDataset(
            model_id: ID!
            dataset_config: DatasetConfigInput!
        ): Model!
        # train model
        train(
            model_id: ID!
        ): Model!
    }
`

