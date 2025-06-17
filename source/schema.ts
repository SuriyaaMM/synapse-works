export const typeDefs = `#graphql
    # ---------- Layers ----------
    # abstract Layer type
    interface Layer {
        id: ID! 
        type: String! # layer name (Linear, Conv2d)
        name: String # optional name
    }
    # LinearLayer type derived from abstract Layer
    type LinearLayer implements Layer {
        id: ID! 
        type: String!   # must be Linear
        name: String    # optional layer name
        in_features: Int!  # input dimensions
        out_features: Int! # output dimensions
    }
    
    # ---------- Layer Config ----------
    # LinearLayerConfig input 
    input LinearLayerConfig {
        in_features: Int!      
        out_features: Int!
        bias: Boolean
        name: String
    }
    # LayerConfig input for collective layers
    input LayerConfig {
        type: String! # Linear, Conv2D
        linear: LinearLayerConfig # optional LinearLayerConfig
    }
    # ---------- Train Config ----------
    type OptimizerConfig {
        lr: Float!
    }
    type TrainConfig {
        epochs: Int!
        batch_size: Int!
        optimizer: String!
        optimizerConfig: OptimizerConfig!
        loss_function: String!
    }
    input OptimizerConfigInput {
        lr: Float!
    }
    input TrainConfigInput {
        epochs: Int!
        batch_size: Int!
        optimizer: String!
        optimizerConfig: OptimizerConfigInput!
        loss_function: String!
    }
    # ---------- Dataset Config ----------
    interface Dataset {
        name: String!
        split_options: [Float]
        shuffle: Boolean
    }
    type MNISTDataset implements Dataset{
        name: String!
        split_options: [Float]
        shuffle: Boolean
        root: String!
        train: Boolean
        download: Boolean
    }
    input MNISTDatasetInput {
        root: String!
        train: Boolean
        download: Boolean
    }
    input DatasetInput {
        name: String!
        split_options: [Float!]
        shuffle: Boolean
        mnist: MNISTDatasetInput
    }
    # ---------- Model ----------
    # Model type
    type Model {
        id: ID!
        name: String!       # model name
        layers: [Layer!]!   # list of layers that model contains
        trainConfig: TrainConfig!
        dataset: Dataset!
    }
    # ---------- Queries ----------
    type Query {
        # get the model by id
        getModel(id: ID!): Model
        # get all models
        getModels: [Model!]!
    }
    # ---------- Mutations ----------
    type Mutation {
        # create model
        createModel(name: String!): Model!
        # append layer to model
        appendLayer(
            modelId: ID!
            layerConfig: LayerConfig!
        ): Model!
        # set's training configuration
        setTrainConfig(
            modelId: ID!
            trainConfig: TrainConfigInput!
        ): Model!
        # set's dataset configuration
        setDataset(
            modelId: ID!
            dataset: DatasetInput!
        )
    }
`

